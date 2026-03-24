from __future__ import annotations

import csv
import json
from pathlib import Path
import re

SLICE_FIELDS = [
    "expected_vehicle_type",
    "expected_split_tag",
    "expected_view",
    "expected_lighting",
    "expected_distance_bucket",
    "expected_occlusion",
    "expected_scene",
]


def normalize_eval_text(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_thai_characters(text):
    return [char for char in normalize_eval_text(text) if "ก" <= char <= "ฮ"]


def parse_optional_int(value):
    normalized = normalize_eval_text(value)
    return int(normalized) if normalized else None


def normalize_box_dict(box):
    if not box:
        return None
    values = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
    if any(value is None or value == "" for value in values):
        return None
    return {
        "x1": int(box["x1"]),
        "y1": int(box["y1"]),
        "x2": int(box["x2"]),
        "y2": int(box["y2"]),
    }


def compute_iou(predicted_box, expected_box):
    if not predicted_box or not expected_box:
        return None

    inter_x1 = max(predicted_box["x1"], expected_box["x1"])
    inter_y1 = max(predicted_box["y1"], expected_box["y1"])
    inter_x2 = min(predicted_box["x2"], expected_box["x2"])
    inter_y2 = min(predicted_box["y2"], expected_box["y2"])

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    predicted_area = (predicted_box["x2"] - predicted_box["x1"]) * (predicted_box["y2"] - predicted_box["y1"])
    expected_area = (expected_box["x2"] - expected_box["x1"]) * (expected_box["y2"] - expected_box["y1"])
    denominator = predicted_area + expected_area - inter_area
    if denominator <= 0:
        return 0.0
    return inter_area / denominator


def levenshtein_distance(left, right):
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion_cost = current_row[right_index - 1] + 1
            deletion_cost = previous_row[right_index] + 1
            substitution_cost = previous_row[right_index - 1] + (left_char != right_char)
            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
        previous_row = current_row
    return previous_row[-1]


def character_error_rate(predicted_text, expected_text):
    normalized_predicted = normalize_eval_text(predicted_text)
    normalized_expected = normalize_eval_text(expected_text)
    if not normalized_expected:
        return 0.0 if not normalized_predicted else 1.0
    return levenshtein_distance(normalized_predicted, normalized_expected) / len(normalized_expected)


def load_ground_truth(csv_path: Path):
    ground_truth = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"image_path", "plate_text", "province"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"ground truth CSV ขาดคอลัมน์ที่จำเป็น: {missing}")

        for row in reader:
            raw_path = row["image_path"].strip()
            resolved_path = Path(raw_path)
            if not resolved_path.is_absolute():
                resolved_path = (csv_path.parent / resolved_path).resolve()

            plate_text = normalize_eval_text(row.get("plate_text", ""))
            province = normalize_eval_text(row.get("province", ""))
            combined_text = normalize_eval_text(row.get("combined_text", "") or f"{plate_text}\n{province}".strip())
            vehicle_type = normalize_eval_text(row.get("vehicle_type", ""))
            bbox = normalize_box_dict(
                {
                    "x1": parse_optional_int(row.get("bbox_x1", "")),
                    "y1": parse_optional_int(row.get("bbox_y1", "")),
                    "x2": parse_optional_int(row.get("bbox_x2", "")),
                    "y2": parse_optional_int(row.get("bbox_y2", "")),
                }
            )
            ground_truth[str(resolved_path)] = {
                "image_path": raw_path,
                "plate_text": plate_text,
                "province": province,
                "combined_text": combined_text,
                "vehicle_type": vehicle_type,
                "split_tag": normalize_eval_text(row.get("split_tag", "")),
                "view": normalize_eval_text(row.get("view", "")),
                "lighting": normalize_eval_text(row.get("lighting", "")),
                "distance_bucket": normalize_eval_text(row.get("distance_bucket", "")),
                "occlusion": normalize_eval_text(row.get("occlusion", "")),
                "scene": normalize_eval_text(row.get("scene", "")),
                "bbox": bbox,
            }
    return ground_truth


def build_thai_character_confusion_rows(evaluation_rows):
    confusion_counts = {}
    for row in evaluation_rows:
        if row.get("evaluation_status") != "evaluated":
            continue

        expected_chars = extract_thai_characters(row.get("expected_plate_text", ""))
        predicted_chars = extract_thai_characters(row.get("predicted_plate_text", ""))
        max_length = max(len(expected_chars), len(predicted_chars))
        for index in range(max_length):
            expected_char = expected_chars[index] if index < len(expected_chars) else "<missing>"
            predicted_char = predicted_chars[index] if index < len(predicted_chars) else "<missing>"
            if expected_char == predicted_char:
                continue

            key = (index + 1, expected_char, predicted_char)
            confusion_counts[key] = confusion_counts.get(key, 0) + 1

    rows = [
        {
            "position": position,
            "expected_char": expected_char,
            "predicted_char": predicted_char,
            "count": count,
        }
        for (position, expected_char, predicted_char), count in sorted(
            confusion_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2])
        )
    ]
    return rows


def build_group_accuracy_rows(evaluation_rows, group_field):
    grouped = {}
    for row in evaluation_rows:
        if row.get("evaluation_status") != "evaluated":
            continue
        group_key = row.get(group_field) or "<unknown>"
        bucket = grouped.setdefault(
            group_key,
            {
                "samples": 0,
                "success_predictions": 0,
                "low_confidence_predictions": 0,
                "failed_predictions": 0,
                "plate_exact_matches": 0,
                "province_exact_matches": 0,
                "combined_exact_matches": 0,
                "mean_plate_cer_sum": 0.0,
                "mean_province_cer_sum": 0.0,
                "mean_combined_cer_sum": 0.0,
                "iou_sum": 0.0,
                "iou_count": 0,
                "iou_at_50_hits": 0,
            },
        )
        bucket["samples"] += 1
        bucket["success_predictions"] += int(row.get("status") == "success")
        bucket["low_confidence_predictions"] += int(row.get("status") == "low_confidence")
        bucket["failed_predictions"] += int(row.get("status") == "failed")
        bucket["plate_exact_matches"] += int(bool(row["plate_exact_match"]))
        bucket["province_exact_matches"] += int(bool(row["province_exact_match"]))
        bucket["combined_exact_matches"] += int(bool(row["combined_exact_match"]))
        bucket["mean_plate_cer_sum"] += float(row["plate_cer"])
        bucket["mean_province_cer_sum"] += float(row["province_cer"])
        bucket["mean_combined_cer_sum"] += float(row["combined_cer"])
        if row.get("iou") not in (None, ""):
            bucket["iou_sum"] += float(row["iou"])
            bucket["iou_count"] += 1
            bucket["iou_at_50_hits"] += int(float(row["iou"]) >= 0.5)

    output_rows = []
    for group_key, bucket in sorted(grouped.items()):
        samples = bucket["samples"]
        output_rows.append(
            {
                "group_type": group_field,
                "group_value": group_key,
                "samples": samples,
                "success_rate": round(bucket["success_predictions"] / samples, 6),
                "low_confidence_rate": round(bucket["low_confidence_predictions"] / samples, 6),
                "failed_rate": round(bucket["failed_predictions"] / samples, 6),
                "plate_exact_accuracy": round(bucket["plate_exact_matches"] / samples, 6),
                "province_exact_accuracy": round(bucket["province_exact_matches"] / samples, 6),
                "combined_exact_accuracy": round(bucket["combined_exact_matches"] / samples, 6),
                "mean_plate_cer": round(bucket["mean_plate_cer_sum"] / samples, 6),
                "mean_province_cer": round(bucket["mean_province_cer_sum"] / samples, 6),
                "mean_combined_cer": round(bucket["mean_combined_cer_sum"] / samples, 6),
                "mean_iou": round(bucket["iou_sum"] / bucket["iou_count"], 6) if bucket["iou_count"] else "",
                "detection_iou_at_0_5": round(bucket["iou_at_50_hits"] / bucket["iou_count"], 6) if bucket["iou_count"] else "",
            }
        )
    return output_rows


def evaluate_predictions(summary_rows, ground_truth_by_path):
    evaluation_rows = []
    matched_paths = set()

    for summary in summary_rows:
        resolved_path = str(Path(summary["image_path"]).resolve())
        ground_truth = ground_truth_by_path.get(resolved_path)
        if ground_truth is None:
            evaluation_rows.append(
                {
                    "image_path": summary["image_path"],
                    "status": summary["status"],
                    "evaluation_status": "missing_ground_truth",
                    "expected_plate_text": "",
                    "predicted_plate_text": summary.get("plate_text", ""),
                    "expected_province": "",
                    "predicted_province": summary.get("province", ""),
                    "expected_combined_text": "",
                    "predicted_combined_text": summary.get("combined_text", ""),
                    "plate_exact_match": False,
                    "province_exact_match": False,
                    "combined_exact_match": False,
                    "plate_cer": "",
                    "province_cer": "",
                    "combined_cer": "",
                }
            )
            continue

        matched_paths.add(resolved_path)
        predicted_plate = normalize_eval_text(summary.get("plate_text", ""))
        predicted_province = normalize_eval_text(summary.get("province", ""))
        predicted_combined = normalize_eval_text(summary.get("combined_text", "") or f"{predicted_plate}\n{predicted_province}".strip())
        predicted_box = normalize_box_dict(summary.get("box"))

        expected_plate = ground_truth["plate_text"]
        expected_province = ground_truth["province"]
        expected_combined = ground_truth["combined_text"]
        expected_vehicle_type = ground_truth.get("vehicle_type", "")
        expected_box = ground_truth.get("bbox")
        iou = compute_iou(predicted_box, expected_box)

        evaluation_rows.append(
            {
                "image_path": summary["image_path"],
                "status": summary["status"],
                "evaluation_status": "evaluated",
                "expected_vehicle_type": expected_vehicle_type,
                "predicted_vehicle_type": summary.get("vehicle_type", ""),
                "expected_split_tag": ground_truth.get("split_tag", ""),
                "expected_view": ground_truth.get("view", ""),
                "expected_lighting": ground_truth.get("lighting", ""),
                "expected_distance_bucket": ground_truth.get("distance_bucket", ""),
                "expected_occlusion": ground_truth.get("occlusion", ""),
                "expected_scene": ground_truth.get("scene", ""),
                "expected_plate_text": expected_plate,
                "predicted_plate_text": predicted_plate,
                "expected_province": expected_province,
                "predicted_province": predicted_province,
                "expected_combined_text": expected_combined,
                "predicted_combined_text": predicted_combined,
                "expected_bbox": expected_box,
                "predicted_bbox": predicted_box,
                "iou": round(iou, 6) if iou is not None else "",
                "detection_iou_at_0_5": iou is not None and iou >= 0.5,
                "plate_exact_match": predicted_plate == expected_plate,
                "province_exact_match": predicted_province == expected_province,
                "combined_exact_match": predicted_combined == expected_combined,
                "plate_cer": round(character_error_rate(predicted_plate, expected_plate), 6),
                "province_cer": round(character_error_rate(predicted_province, expected_province), 6),
                "combined_cer": round(character_error_rate(predicted_combined, expected_combined), 6),
            }
        )

    for resolved_path, ground_truth in ground_truth_by_path.items():
        if resolved_path in matched_paths:
            continue
        evaluation_rows.append(
            {
                "image_path": ground_truth["image_path"],
                "status": "failed",
                "evaluation_status": "missing_prediction",
                "expected_vehicle_type": ground_truth.get("vehicle_type", ""),
                "predicted_vehicle_type": "",
                "expected_split_tag": ground_truth.get("split_tag", ""),
                "expected_view": ground_truth.get("view", ""),
                "expected_lighting": ground_truth.get("lighting", ""),
                "expected_distance_bucket": ground_truth.get("distance_bucket", ""),
                "expected_occlusion": ground_truth.get("occlusion", ""),
                "expected_scene": ground_truth.get("scene", ""),
                "expected_plate_text": ground_truth["plate_text"],
                "predicted_plate_text": "",
                "expected_province": ground_truth["province"],
                "predicted_province": "",
                "expected_combined_text": ground_truth["combined_text"],
                "predicted_combined_text": "",
                "expected_bbox": ground_truth.get("bbox"),
                "predicted_bbox": None,
                "iou": "",
                "detection_iou_at_0_5": False,
                "plate_exact_match": False,
                "province_exact_match": False,
                "combined_exact_match": False,
                "plate_cer": round(character_error_rate("", ground_truth["plate_text"]), 6),
                "province_cer": round(character_error_rate("", ground_truth["province"]), 6),
                "combined_cer": round(character_error_rate("", ground_truth["combined_text"]), 6),
            }
        )

    evaluated_rows = [row for row in evaluation_rows if row["evaluation_status"] == "evaluated"]
    aggregate = {
        "total_ground_truth_rows": len(ground_truth_by_path),
        "total_predictions": len(summary_rows),
        "evaluated_rows": len(evaluated_rows),
        "missing_ground_truth_rows": sum(row["evaluation_status"] == "missing_ground_truth" for row in evaluation_rows),
        "missing_prediction_rows": sum(row["evaluation_status"] == "missing_prediction" for row in evaluation_rows),
        "success_rate": 0.0,
        "low_confidence_rate": 0.0,
        "failed_rate": 0.0,
        "plate_exact_accuracy": 0.0,
        "province_exact_accuracy": 0.0,
        "combined_exact_accuracy": 0.0,
        "mean_plate_cer": 0.0,
        "mean_province_cer": 0.0,
        "mean_combined_cer": 0.0,
        "mean_iou": 0.0,
        "detection_iou_at_0_5": 0.0,
    }
    if evaluated_rows:
        aggregate["success_rate"] = round(sum(row["status"] == "success" for row in evaluated_rows) / len(evaluated_rows), 6)
        aggregate["low_confidence_rate"] = round(sum(row["status"] == "low_confidence" for row in evaluated_rows) / len(evaluated_rows), 6)
        aggregate["failed_rate"] = round(sum(row["status"] == "failed" for row in evaluated_rows) / len(evaluated_rows), 6)
        aggregate["plate_exact_accuracy"] = round(sum(row["plate_exact_match"] for row in evaluated_rows) / len(evaluated_rows), 6)
        aggregate["province_exact_accuracy"] = round(sum(row["province_exact_match"] for row in evaluated_rows) / len(evaluated_rows), 6)
        aggregate["combined_exact_accuracy"] = round(sum(row["combined_exact_match"] for row in evaluated_rows) / len(evaluated_rows), 6)
        aggregate["mean_plate_cer"] = round(sum(row["plate_cer"] for row in evaluated_rows) / len(evaluated_rows), 6)
        aggregate["mean_province_cer"] = round(sum(row["province_cer"] for row in evaluated_rows) / len(evaluated_rows), 6)
        aggregate["mean_combined_cer"] = round(sum(row["combined_cer"] for row in evaluated_rows) / len(evaluated_rows), 6)

        rows_with_iou = [row for row in evaluated_rows if row["iou"] not in (None, "")]
        if rows_with_iou:
            aggregate["mean_iou"] = round(sum(float(row["iou"]) for row in rows_with_iou) / len(rows_with_iou), 6)
            aggregate["detection_iou_at_0_5"] = round(sum(bool(row["detection_iou_at_0_5"]) for row in rows_with_iou) / len(rows_with_iou), 6)

    return aggregate, evaluation_rows


def write_evaluation_reports(aggregate, evaluation_rows, output_dir: Path):
    json_path = output_dir / "evaluation_report.json"
    csv_path = output_dir / "evaluation_report.csv"
    leaderboard_json_path = output_dir / "evaluation_leaderboard.json"
    leaderboard_csv_path = output_dir / "evaluation_leaderboard.csv"
    confusion_json_path = output_dir / "thai_char_confusion_report.json"
    confusion_csv_path = output_dir / "thai_char_confusion_report.csv"

    leaderboard_groups = {
        field: build_group_accuracy_rows(evaluation_rows, field)
        for field in ["expected_province", *SLICE_FIELDS]
        if any(row.get(field) for row in evaluation_rows)
    }
    thai_char_confusions = build_thai_character_confusion_rows(evaluation_rows)
    payload = {
        "aggregate": aggregate,
        "rows": evaluation_rows,
        "leaderboards": leaderboard_groups,
        "thai_char_confusions": thai_char_confusions,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "image_path",
        "status",
        "evaluation_status",
        "expected_vehicle_type",
        "predicted_vehicle_type",
        "expected_split_tag",
        "expected_view",
        "expected_lighting",
        "expected_distance_bucket",
        "expected_occlusion",
        "expected_scene",
        "expected_plate_text",
        "predicted_plate_text",
        "expected_province",
        "predicted_province",
        "expected_combined_text",
        "predicted_combined_text",
        "expected_bbox",
        "predicted_bbox",
        "iou",
        "detection_iou_at_0_5",
        "plate_exact_match",
        "province_exact_match",
        "combined_exact_match",
        "plate_cer",
        "province_cer",
        "combined_cer",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in evaluation_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    leaderboard_rows = [row for rows in leaderboard_groups.values() for row in rows]
    leaderboard_json_path.write_text(json.dumps(leaderboard_groups, ensure_ascii=False, indent=2), encoding="utf-8")
    leaderboard_fields = [
        "group_type",
        "group_value",
        "samples",
        "success_rate",
        "low_confidence_rate",
        "failed_rate",
        "plate_exact_accuracy",
        "province_exact_accuracy",
        "combined_exact_accuracy",
        "mean_plate_cer",
        "mean_province_cer",
        "mean_combined_cer",
        "mean_iou",
        "detection_iou_at_0_5",
    ]
    with leaderboard_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=leaderboard_fields)
        writer.writeheader()
        for row in leaderboard_rows:
            writer.writerow({key: row.get(key, "") for key in leaderboard_fields})

    confusion_json_path.write_text(json.dumps({"rows": thai_char_confusions}, ensure_ascii=False, indent=2), encoding="utf-8")
    confusion_fields = ["position", "expected_char", "predicted_char", "count"]
    with confusion_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=confusion_fields)
        writer.writeheader()
        for row in thai_char_confusions:
            writer.writerow({key: row.get(key, "") for key in confusion_fields})

    return {
        "evaluation_json": json_path,
        "evaluation_csv": csv_path,
        "leaderboard_json": leaderboard_json_path,
        "leaderboard_csv": leaderboard_csv_path,
        "confusion_json": confusion_json_path,
        "confusion_csv": confusion_csv_path,
    }
