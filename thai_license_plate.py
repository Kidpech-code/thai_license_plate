import argparse
import csv
import json
from dataclasses import dataclass
from dataclasses import asdict
from difflib import get_close_matches
from itertools import product
import logging
from pathlib import Path
import re
import sys
from typing import Iterable

import cv2
import easyocr
from ultralytics import YOLOWorld

MODEL_PATH = "yolov8s-worldv2.pt"
IMAGE_PATH = "car_image.jpeg"
CONFIG_PATH = "plate_config.json"
OUTPUT_BASENAME = "plate_result"
DEFAULT_VEHICLE_TYPE = "auto"
VEHICLE_TYPE_CHOICES = ["auto", "any", "private_car", "private_pickup", "private_van", "taxi"]


@dataclass(frozen=True)
class DomainConfig:
    prompts: list[str]
    confidence_threshold: float
    image_size: int
    max_results: int
    lower_roi_y_ratio: float
    lower_roi_x_ratio: float
    padding_ratio: float
    green_plate_threshold: float
    blue_plate_threshold: float
    valid_image_extensions: tuple[str, ...]
    thai_provinces: tuple[str, ...]
    series_prefixes_by_vehicle_type: dict[str, set[str]]
    thai_plate_char_confusions: dict[str, tuple[str, ...]]


@dataclass
class DetectionCandidate:
    source: str
    prompt: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class PlateResult:
    source: str
    prompt: str
    confidence: float
    box: tuple[int, int, int, int]
    vehicle_type: str
    plate_line: str
    province_line: str
    score: float


@dataclass(frozen=True)
class AppConfig:
    image_path: Path
    input_dir: Path | None
    ground_truth_csv: Path | None
    model_path: str
    config_path: Path
    output_dir: Path
    output_basename: str
    save_debug: bool
    vehicle_type: str
    recursive: bool
    log_level: str


def setup_logging(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def load_domain_config(config_path: Path):
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    confusion_map = {}
    for group in raw["thai_character_confusion_groups"]:
        normalized_group = tuple(dict.fromkeys(group))
        for char in normalized_group:
            confusion_map[char] = normalized_group

    series_prefixes = {
        key: set(value)
        for key, value in raw["series_prefixes_by_vehicle_type"].items()
    }

    return DomainConfig(
        prompts=list(raw["prompts"]),
        confidence_threshold=float(raw["confidence_threshold"]),
        image_size=int(raw["image_size"]),
        max_results=int(raw["max_results"]),
        lower_roi_y_ratio=float(raw["lower_roi_y_ratio"]),
        lower_roi_x_ratio=float(raw["lower_roi_x_ratio"]),
        padding_ratio=float(raw["padding_ratio"]),
        green_plate_threshold=float(raw["green_plate_threshold"]),
        blue_plate_threshold=float(raw["blue_plate_threshold"]),
        valid_image_extensions=tuple(ext.lower() for ext in raw["valid_image_extensions"]),
        thai_provinces=tuple(raw["thai_provinces"]),
        series_prefixes_by_vehicle_type=series_prefixes,
        thai_plate_char_confusions=confusion_map,
    )


def load_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพหรือเปิดไฟล์ไม่ได้: {image_path}")
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Thai license plate detection and OCR")
    parser.add_argument("--image", default=IMAGE_PATH, help="Path to the input image")
    parser.add_argument("--input-dir", help="Directory of images for batch processing")
    parser.add_argument("--ground-truth-csv", help="CSV file for evaluation with columns: image_path, plate_text, province")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the YOLO-World model")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to the domain config JSON")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--output-basename", default=OUTPUT_BASENAME, help="Base name for output files")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --input-dir for images")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable writing debug JSON with all candidate results",
    )
    parser.add_argument(
        "--vehicle-type",
        default=DEFAULT_VEHICLE_TYPE,
        choices=VEHICLE_TYPE_CHOICES,
        help="Vehicle type used to score valid Thai plate series prefixes",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()
    return AppConfig(
        image_path=Path(args.image),
        input_dir=Path(args.input_dir) if args.input_dir else None,
        ground_truth_csv=Path(args.ground_truth_csv) if args.ground_truth_csv else None,
        model_path=args.model,
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        output_basename=args.output_basename,
        save_debug=not args.no_debug,
        vehicle_type=args.vehicle_type,
        recursive=args.recursive,
        log_level=args.log_level,
    )


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def expand_box(box, width, height, padding_ratio=0.08):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def is_plate_like(box, image_width, image_height):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return False

    aspect_ratio = box_width / box_height
    relative_area = (box_width * box_height) / (image_width * image_height)
    return 1.5 <= aspect_ratio <= 8.0 and 0.0005 <= relative_area <= 0.08


def preprocess_plate(cropped_plate):
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(enlarged, (5, 5), 0)
    otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        enlarged,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return enlarged, otsu, adaptive


def build_output_paths(config: AppConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = config.output_dir / f"{config.output_basename}_annotated{config.image_path.suffix}"
    crop_path = config.output_dir / f"{config.output_basename}_crop{config.image_path.suffix}"
    json_path = config.output_dir / f"{config.output_basename}.json"
    debug_path = config.output_dir / f"{config.output_basename}_debug.json"
    return annotated_path, crop_path, json_path, debug_path


def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())


def extract_plate_letters(text):
    return "".join(char for char in text if "ก" <= char <= "ฮ")


def get_confusion_options(char, domain_config: DomainConfig):
    return domain_config.thai_plate_char_confusions.get(char, (char,))


def score_confusion_resolution(candidate, raw_letters, domain_config: DomainConfig):
    score = 0.0
    for raw_char, resolved_char in zip(raw_letters, candidate):
        options = get_confusion_options(raw_char, domain_config)
        if resolved_char not in options:
            continue

        option_rank = options.index(resolved_char)
        score += max(0.0, 1.5 - (0.2 * option_rank))
        if resolved_char == raw_char:
            score += 0.1

    return score


def generate_plate_letter_candidates(raw_letters, domain_config: DomainConfig, max_candidates=32):
    if not raw_letters:
        return []

    limited_letters = raw_letters[:3]
    option_groups = [get_confusion_options(char, domain_config) for char in limited_letters]
    expanded_candidates = []
    seen = set()
    for combination in product(*option_groups):
        candidate = "".join(combination)
        if candidate not in seen:
            seen.add(candidate)
            expanded_candidates.append(candidate)
        if len(expanded_candidates) >= max_candidates:
            break

    return expanded_candidates or [limited_letters]


def extract_series_prefix_digit(text, serial_digits):
    if not text:
        return ""

    prefix_match = re.match(r"\s*(\d)", text)
    if not prefix_match:
        return ""

    prefix_digit = prefix_match.group(1)
    if serial_digits and text.strip().endswith(serial_digits) and prefix_digit == serial_digits[0]:
        return ""
    return prefix_digit


def score_letter_candidate(candidate, raw_letters, vehicle_type, domain_config: DomainConfig):
    score = 0.0
    if len(candidate) in (2, 3):
        score += 2.0
    if len(candidate) == 2:
        score += 1.0

    score += score_confusion_resolution(candidate, raw_letters, domain_config)

    allowed_prefixes = domain_config.series_prefixes_by_vehicle_type[vehicle_type]
    if allowed_prefixes and candidate and candidate[0] in allowed_prefixes:
        score += 3.0
    elif candidate and candidate[0] in domain_config.series_prefixes_by_vehicle_type["private_car"] | domain_config.series_prefixes_by_vehicle_type["private_pickup"] | domain_config.series_prefixes_by_vehicle_type["private_van"] | domain_config.series_prefixes_by_vehicle_type["taxi"]:
        score += 1.5

    return score


def normalize_plate_line(text_candidates, vehicle_type, domain_config: DomainConfig):
    best_candidate = ""
    best_score = -1

    for candidate in text_candidates:
        cleaned = clean_text(candidate)
        digit_runs = re.findall(r"\d{1,4}", cleaned)
        digits = max(digit_runs, key=len) if digit_runs else ""
        prefix_digit = extract_series_prefix_digit(cleaned, digits)
        raw_letters = extract_plate_letters(cleaned)

        for letters in generate_plate_letter_candidates(raw_letters, domain_config):
            score = score_letter_candidate(letters, raw_letters, vehicle_type, domain_config)
            if digits:
                score += 2.0
                if len(digits) == 4:
                    score += 1.0
            if prefix_digit:
                score += 1.0
            if len(cleaned) >= 6:
                score += 0.5

            formatted_plate = f"{prefix_digit}{letters} {digits}".strip()
            if score > best_score:
                best_score = score
                best_candidate = formatted_plate

    return best_candidate


def normalize_province_line(text_candidates, domain_config: DomainConfig):
    cleaned_candidates = [clean_text(candidate).replace(" ", "") for candidate in text_candidates if clean_text(candidate)]
    if not cleaned_candidates:
        return ""

    for candidate in cleaned_candidates:
        if candidate in domain_config.thai_provinces:
            return candidate

    for candidate in cleaned_candidates:
        matches = get_close_matches(candidate, domain_config.thai_provinces, n=1, cutoff=0.5)
        if matches:
            return matches[0]

    return cleaned_candidates[0]


def read_plate_lines(reader, cropped_plate, vehicle_type, domain_config: DomainConfig):
    enlarged, otsu, adaptive = preprocess_plate(cropped_plate)

    top_slice = slice(0, int(enlarged.shape[0] * 0.62))
    bottom_slice = slice(int(enlarged.shape[0] * 0.55), enlarged.shape[0])

    top_variants = [
        enlarged[top_slice, :],
        otsu[top_slice, :],
        adaptive[top_slice, :],
    ]
    bottom_variants = [
        enlarged[bottom_slice, :],
        otsu[bottom_slice, :],
        adaptive[bottom_slice, :],
        enlarged,
        otsu,
    ]

    upper_candidates = []
    lower_candidates = []
    for variant in top_variants:
        upper_candidates.extend(reader.readtext(variant, detail=0, paragraph=False))
    for variant in bottom_variants:
        lower_candidates.extend(reader.readtext(variant, detail=0, paragraph=False))

    plate_line = normalize_plate_line(upper_candidates, vehicle_type, domain_config)
    province_line = normalize_province_line(lower_candidates, domain_config)
    return plate_line, province_line


def score_plate_result(plate_line, province_line, domain_config: DomainConfig):
    score = 0.0
    if re.fullmatch(r"[ก-ฮ]{2} \d{4}", plate_line):
        score += 4.0
    elif re.search(r"\d{4}", plate_line):
        score += 2.0

    if province_line in domain_config.thai_provinces:
        score += 4.0
    elif province_line:
        score += 1.0

    if province_line == "กรุงเทพมหานคร":
        score += 1.5

    return score


def annotate_and_save(image, best_result, output_path: Path):
    annotated = image.copy()
    x1, y1, x2, y2 = best_result.box
    label = f"{best_result.plate_line} | {best_result.province_line}"
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        annotated,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), annotated)


def save_crop(image, box, output_path: Path):
    x1, y1, x2, y2 = box
    cropped = image[y1:y2, x1:x2]
    cv2.imwrite(str(output_path), cropped)


def write_result_json(best_result: PlateResult, image_path: Path, output_path: Path):
    payload = {
        "image_path": str(image_path),
        "plate_text": best_result.plate_line,
        "province": best_result.province_line,
        "combined_text": f"{best_result.plate_line}\n{best_result.province_line}".strip(),
        "source": best_result.source,
        "prompt": best_result.prompt,
        "vehicle_type": best_result.vehicle_type,
        "confidence": round(best_result.confidence, 6),
        "score": round(best_result.score, 6),
        "box": {
            "x1": best_result.box[0],
            "y1": best_result.box[1],
            "x2": best_result.box[2],
            "y2": best_result.box[3],
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_debug_json(results, merged_candidates, output_path: Path):
    payload = {
        "candidate_count": len(merged_candidates),
        "merged_candidates": [asdict(candidate) for candidate in merged_candidates],
        "ranked_results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_files(summary_rows, output_dir: Path):
    jsonl_path = output_dir / "results_summary.jsonl"
    csv_path = output_dir / "results_summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for row in summary_rows:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "image_path",
        "status",
        "vehicle_type",
        "plate_text",
        "province",
        "combined_text",
        "source",
        "prompt",
        "confidence",
        "score",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
                "status": "missing_prediction",
                "evaluation_status": "missing_prediction",
                "expected_vehicle_type": ground_truth.get("vehicle_type", ""),
                "predicted_vehicle_type": "",
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

    per_province = build_group_accuracy_rows(evaluation_rows, "expected_province")
    per_vehicle_type = build_group_accuracy_rows(evaluation_rows, "expected_vehicle_type")
    thai_char_confusions = build_thai_character_confusion_rows(evaluation_rows)
    payload = {
        "aggregate": aggregate,
        "rows": evaluation_rows,
        "leaderboards": {
            "per_province": per_province,
            "per_vehicle_type": per_vehicle_type,
        },
        "thai_char_confusions": thai_char_confusions,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "image_path",
        "status",
        "evaluation_status",
        "expected_vehicle_type",
        "predicted_vehicle_type",
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

    leaderboard_rows = per_province + per_vehicle_type
    leaderboard_payload = {
        "per_province": per_province,
        "per_vehicle_type": per_vehicle_type,
    }
    leaderboard_json_path.write_text(json.dumps(leaderboard_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    leaderboard_fields = [
        "group_type",
        "group_value",
        "samples",
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


def is_valid_image(path: Path, domain_config: DomainConfig):
    return path.is_file() and path.suffix.lower() in domain_config.valid_image_extensions


def iter_input_images(config: AppConfig, domain_config: DomainConfig) -> Iterable[Path]:
    if config.input_dir is not None:
        iterator = config.input_dir.rglob("*") if config.recursive else config.input_dir.iterdir()
        for path in sorted(iterator):
            if is_valid_image(path, domain_config):
                yield path
        return

    yield config.image_path


class PlateRecognizer:
    def __init__(self, config: AppConfig, domain_config: DomainConfig):
        self.config = config
        self.domain_config = domain_config
        logging.info("กำลังโหลดโมเดล: %s", config.model_path)
        self.model = YOLOWorld(config.model_path)
        logging.info("กำลังโหลด OCR: ภาษาไทย + อังกฤษ")
        self.reader = easyocr.Reader(["th", "en"])

    def infer_vehicle_type_from_plate(self, cropped_plate, requested_vehicle_type):
        if requested_vehicle_type != "auto":
            return requested_vehicle_type

        hsv = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 25, 25), (100, 255, 255))
        blue_mask = cv2.inRange(hsv, (90, 40, 40), (135, 255, 255))
        total_pixels = max(cropped_plate.shape[0] * cropped_plate.shape[1], 1)
        green_ratio = cv2.countNonZero(green_mask) / total_pixels
        blue_ratio = cv2.countNonZero(blue_mask) / total_pixels

        if green_ratio >= self.domain_config.green_plate_threshold:
            return "private_pickup"
        if blue_ratio >= self.domain_config.blue_plate_threshold:
            return "private_van"
        return "private_car"

    def find_plate_by_contours(self, image):
        image_height, image_width = image.shape[:2]
        x_start = int(image_width * self.domain_config.lower_roi_x_ratio)
        x_end = int(image_width * (1 - self.domain_config.lower_roi_x_ratio))
        y_start = int(image_height * self.domain_config.lower_roi_y_ratio)
        y_end = int(image_height * 0.95)
        roi = image[y_start:y_end, x_start:x_end]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 60, 180)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 25, 25), (100, 255, 255))
        combined = cv2.bitwise_or(edges, green_mask)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            global_box = (x + x_start, y + y_start, x + x_start + width, y + y_start + height)
            if not is_plate_like(global_box, image_width, image_height):
                continue

            area = width * height
            if area < 2000:
                continue

            green_ratio = cv2.countNonZero(green_mask[y:y + height, x:x + width]) / max(area, 1)
            contour_area = cv2.contourArea(contour)
            extent = contour_area / max(area, 1)
            center_x = global_box[0] + width / 2
            center_y = global_box[1] + height / 2
            center_score = 1.0 - (abs(center_x - image_width / 2) / (image_width / 2))
            vertical_score = center_y / image_height
            score = green_ratio * 3.5 + extent + center_score + vertical_score

            candidates.append(
                DetectionCandidate(
                    source="contour",
                    prompt="contour fallback",
                    confidence=min(0.99, score / 5),
                    box=global_box,
                )
            )

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[: self.domain_config.max_results]

    def collect_yolo_detections(self, image):
        image_height, image_width = image.shape[:2]
        detections = []

        for prompt in self.domain_config.prompts:
            self.model.set_classes([prompt, ""])
            results = self.model.predict(
                image,
                conf=self.domain_config.confidence_threshold,
                imgsz=self.domain_config.image_size,
                verbose=False,
            )

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), image_width, image_height)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if not is_plate_like((x1, y1, x2, y2), image_width, image_height):
                        continue
                    detections.append(
                        DetectionCandidate(
                            source="yolo",
                            prompt=prompt,
                            confidence=float(box.conf[0]),
                            box=(x1, y1, x2, y2),
                        )
                    )

        return detections

    def merge_candidates(self, candidates):
        if not candidates:
            return []

        boxes = []
        confidences = []
        for candidate in candidates:
            x1, y1, x2, y2 = candidate.box
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(candidate.confidence)

        selected_indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.domain_config.confidence_threshold,
            0.3,
        )
        if len(selected_indexes) == 0:
            return []

        flattened_indexes = [int(index) for index in selected_indexes.flatten()]
        filtered = [candidates[index] for index in flattened_indexes]
        filtered.sort(key=lambda item: item.confidence, reverse=True)
        return filtered[: self.domain_config.max_results]

    def process_image(self, image_path: Path):
        logging.info("กำลังอ่านภาพ: %s", image_path)
        image = load_image(image_path)
        logging.info("กำลังค้นหาป้ายทะเบียนด้วย prompts: %s", ", ".join(self.domain_config.prompts))
        yolo_candidates = self.collect_yolo_detections(image)
        contour_candidates = self.find_plate_by_contours(image)
        merged_candidates = self.merge_candidates(yolo_candidates + contour_candidates)

        if not merged_candidates:
            raise RuntimeError("ไม่พบป้ายทะเบียนในภาพนี้")

        results = []
        for candidate in merged_candidates:
            expanded_box = expand_box(
                candidate.box,
                image.shape[1],
                image.shape[0],
                padding_ratio=self.domain_config.padding_ratio,
            )
            x1, y1, x2, y2 = expanded_box
            cropped_plate = image[y1:y2, x1:x2]
            resolved_vehicle_type = self.infer_vehicle_type_from_plate(cropped_plate, self.config.vehicle_type)
            plate_line, province_line = read_plate_lines(
                self.reader,
                cropped_plate,
                resolved_vehicle_type,
                self.domain_config,
            )
            score = score_plate_result(plate_line, province_line, self.domain_config) + candidate.confidence
            results.append(
                PlateResult(
                    source=candidate.source,
                    prompt=candidate.prompt,
                    confidence=candidate.confidence,
                    box=expanded_box,
                    vehicle_type=resolved_vehicle_type,
                    plate_line=plate_line,
                    province_line=province_line,
                    score=score,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return image, merged_candidates, results, results[0]


def main():
    config = parse_args()
    setup_logging(config.log_level)
    domain_config = load_domain_config(config.config_path)
    recognizer = PlateRecognizer(config, domain_config)
    summary_rows = []

    exit_code = 0
    for image_path in iter_input_images(config, domain_config):
        image_specific_config = AppConfig(
            image_path=image_path,
            input_dir=config.input_dir,
            ground_truth_csv=config.ground_truth_csv,
            model_path=config.model_path,
            config_path=config.config_path,
            output_dir=config.output_dir,
            output_basename=f"{config.output_basename}_{image_path.stem}" if config.input_dir else config.output_basename,
            save_debug=config.save_debug,
            vehicle_type=config.vehicle_type,
            recursive=config.recursive,
            log_level=config.log_level,
        )
        annotated_path, crop_path, result_json_path, debug_json_path = build_output_paths(image_specific_config)

        try:
            image, merged_candidates, results, best_result = recognizer.process_image(image_path)
            logging.info("--------------------------------------------------")
            logging.info("แหล่งที่มาของกรอบ: %s", best_result.source)
            logging.info("Prompt ที่เลือก: %s", best_result.prompt)
            logging.info("Vehicle type ที่ใช้: %s", best_result.vehicle_type)
            logging.info("Confidence: %.4f", best_result.confidence)
            logging.info(
                "พิกัดป้ายทะเบียน: %s, %s, %s, %s",
                best_result.box[0],
                best_result.box[1],
                best_result.box[2],
                best_result.box[3],
            )
            logging.info("ผลลัพธ์สุดท้าย")
            logging.info(best_result.plate_line or "อ่านทะเบียนไม่ออก")
            logging.info(best_result.province_line or "อ่านจังหวัดไม่ออก")

            annotate_and_save(image, best_result, annotated_path)
            save_crop(image, best_result.box, crop_path)
            write_result_json(best_result, image_path, result_json_path)
            if image_specific_config.save_debug:
                write_debug_json(results, merged_candidates, debug_json_path)

            summary_rows.append(
                {
                    "image_path": str(image_path),
                    "status": "success",
                    "vehicle_type": best_result.vehicle_type,
                    "plate_text": best_result.plate_line,
                    "province": best_result.province_line,
                    "combined_text": f"{best_result.plate_line}\n{best_result.province_line}".strip(),
                    "source": best_result.source,
                    "prompt": best_result.prompt,
                    "confidence": round(best_result.confidence, 6),
                    "score": round(best_result.score, 6),
                    "box": {
                        "x1": best_result.box[0],
                        "y1": best_result.box[1],
                        "x2": best_result.box[2],
                        "y2": best_result.box[3],
                    },
                    "error": "",
                }
            )
            logging.info("บันทึกภาพผลลัพธ์ไว้ที่: %s", annotated_path)
            logging.info("บันทึกภาพ crop ป้ายไว้ที่: %s", crop_path)
            logging.info("บันทึกผลลัพธ์ JSON ไว้ที่: %s", result_json_path)
            if image_specific_config.save_debug:
                logging.info("บันทึก debug JSON ไว้ที่: %s", debug_json_path)
        except Exception as exc:
            exit_code = 1
            logging.error("ประมวลผลภาพไม่สำเร็จ %s: %s", image_path, exc)
            summary_rows.append(
                {
                    "image_path": str(image_path),
                    "status": "error",
                    "vehicle_type": "",
                    "plate_text": "",
                    "province": "",
                    "combined_text": "",
                    "source": "",
                    "prompt": "",
                    "confidence": "",
                    "score": "",
                    "box": None,
                    "error": str(exc),
                }
            )

    if summary_rows:
        write_summary_files(summary_rows, config.output_dir)
    if config.ground_truth_csv is not None:
        ground_truth_by_path = load_ground_truth(config.ground_truth_csv)
        aggregate, evaluation_rows = evaluate_predictions(summary_rows, ground_truth_by_path)
        report_paths = write_evaluation_reports(aggregate, evaluation_rows, config.output_dir)
        logging.info("Evaluation summary: plate_exact_accuracy=%.4f province_exact_accuracy=%.4f combined_exact_accuracy=%.4f mean_combined_cer=%.4f", aggregate["plate_exact_accuracy"], aggregate["province_exact_accuracy"], aggregate["combined_exact_accuracy"], aggregate["mean_combined_cer"])
        logging.info("Detection summary: mean_iou=%.4f detection_iou_at_0_5=%.4f", aggregate["mean_iou"], aggregate["detection_iou_at_0_5"])
        logging.info("บันทึก evaluation JSON ไว้ที่: %s", report_paths["evaluation_json"])
        logging.info("บันทึก evaluation CSV ไว้ที่: %s", report_paths["evaluation_csv"])
        logging.info("บันทึก leaderboard JSON ไว้ที่: %s", report_paths["leaderboard_json"])
        logging.info("บันทึก leaderboard CSV ไว้ที่: %s", report_paths["leaderboard_csv"])
        logging.info("บันทึก Thai confusion JSON ไว้ที่: %s", report_paths["confusion_json"])
        logging.info("บันทึก Thai confusion CSV ไว้ที่: %s", report_paths["confusion_csv"])
    sys.exit(exit_code)


if __name__ == "__main__":
    main()