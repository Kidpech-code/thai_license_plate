from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from .types import PlateResult


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
        "status": best_result.decision_status,
        "plate_text": best_result.plate_line,
        "province": best_result.province_line,
        "combined_text": f"{best_result.plate_line}\n{best_result.province_line}".strip(),
        "source": best_result.source,
        "prompt": best_result.prompt,
        "vehicle_type": best_result.vehicle_type,
        "confidence": round(best_result.confidence, 6),
        "score": round(best_result.score, 6),
        "text_score": round(best_result.text_score, 6),
        "decision_reasons": best_result.decision_reasons,
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
        "text_score",
        "decision_reasons",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            normalized = dict(row)
            if isinstance(normalized.get("decision_reasons"), list):
                normalized["decision_reasons"] = ",".join(normalized["decision_reasons"])
            writer.writerow({key: normalized.get(key, "") for key in fieldnames})
