from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .types import AppConfig, DomainConfig

MODEL_PATH = "yolov8s-worldv2.pt"
IMAGE_PATH = "car_image.jpg"
CONFIG_PATH = "plate_config.json"
OUTPUT_BASENAME = "plate_result"
DEFAULT_VEHICLE_TYPE = "auto"
VEHICLE_TYPE_CHOICES = ["auto", "any", "private_car", "private_pickup", "private_van", "taxi"]
JPEG_SUFFIX_ALIASES = {
    ".jpg": ".jpeg",
    ".jpeg": ".jpg",
}


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
        key: tuple(dict.fromkeys(value))
        for key, value in raw["series_prefixes_by_vehicle_type"].items()
    }
    valid_two_letter_series = {
        key: tuple(dict.fromkeys(value))
        for key, value in raw.get("valid_two_letter_series_by_vehicle_type", {}).items()
    }

    prompt_batches = raw.get("prompt_batches") or [raw["prompts"]]
    normalized_prompt_batches = []
    for batch in prompt_batches:
        deduplicated = tuple(dict.fromkeys(batch))
        if deduplicated:
            normalized_prompt_batches.append(deduplicated)

    prompts = tuple(dict.fromkeys(prompt for batch in normalized_prompt_batches for prompt in batch))

    return DomainConfig(
        prompts=prompts,
        prompt_batches=tuple(normalized_prompt_batches),
        confidence_threshold=float(raw["confidence_threshold"]),
        image_size=int(raw["image_size"]),
        max_results=int(raw["max_results"]),
        lower_roi_y_ratio=float(raw["lower_roi_y_ratio"]),
        lower_roi_x_ratio=float(raw["lower_roi_x_ratio"]),
        padding_ratio=float(raw["padding_ratio"]),
        green_plate_threshold=float(raw["green_plate_threshold"]),
        blue_plate_threshold=float(raw["blue_plate_threshold"]),
        success_score_threshold=float(raw.get("success_score_threshold", 8.0)),
        low_confidence_score_threshold=float(raw.get("low_confidence_score_threshold", 4.5)),
        yolo_early_acceptance_confidence=float(raw.get("yolo_early_acceptance_confidence", 0.2)),
        valid_image_extensions=tuple(ext.lower() for ext in raw["valid_image_extensions"]),
        thai_provinces=tuple(raw["thai_provinces"]),
        series_prefixes_by_vehicle_type=series_prefixes,
        valid_two_letter_series_by_vehicle_type={
            vehicle_type: valid_two_letter_series.get(vehicle_type, ())
            for vehicle_type in series_prefixes
        },
        thai_plate_char_confusions=confusion_map,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Thai license plate detection and OCR")
    parser.add_argument("--image", default=IMAGE_PATH, help="Path to the input image")
    parser.add_argument("--input-dir", help="Directory of images for batch processing")
    parser.add_argument(
        "--ground-truth-csv",
        help=(
            "CSV file for evaluation with columns: image_path, plate_text, province and optional "
            "slice fields such as split_tag, view, lighting, distance_bucket, occlusion, scene"
        ),
    )
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


def build_output_paths(config: AppConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = config.output_dir / f"{config.output_basename}_annotated{config.image_path.suffix}"
    crop_path = config.output_dir / f"{config.output_basename}_crop{config.image_path.suffix}"
    json_path = config.output_dir / f"{config.output_basename}.json"
    debug_path = config.output_dir / f"{config.output_basename}_debug.json"
    return annotated_path, crop_path, json_path, debug_path


def is_valid_image(path: Path, domain_config: DomainConfig):
    return path.is_file() and path.suffix.lower() in domain_config.valid_image_extensions


def resolve_image_path(image_path: Path):
    if image_path.is_file():
        return image_path

    alternate_suffix = JPEG_SUFFIX_ALIASES.get(image_path.suffix.lower())
    if alternate_suffix is None:
        return image_path

    alternate_path = image_path.with_suffix(alternate_suffix)
    if alternate_path.is_file():
        return alternate_path

    return image_path


def iter_input_images(config: AppConfig, domain_config: DomainConfig):
    if config.input_dir is not None:
        iterator = config.input_dir.rglob("*") if config.recursive else config.input_dir.iterdir()
        for path in sorted(iterator):
            if is_valid_image(path, domain_config):
                yield path
        return

    yield resolve_image_path(config.image_path)
