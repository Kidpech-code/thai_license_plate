from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DomainConfig:
    prompts: tuple[str, ...]
    prompt_batches: tuple[tuple[str, ...], ...]
    confidence_threshold: float
    image_size: int
    max_results: int
    lower_roi_y_ratio: float
    lower_roi_x_ratio: float
    padding_ratio: float
    green_plate_threshold: float
    blue_plate_threshold: float
    success_score_threshold: float
    low_confidence_score_threshold: float
    yolo_early_acceptance_confidence: float
    valid_image_extensions: tuple[str, ...]
    thai_provinces: tuple[str, ...]
    series_prefixes_by_vehicle_type: dict[str, tuple[str, ...]]
    valid_two_letter_series_by_vehicle_type: dict[str, tuple[str, ...]]
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
    decision_status: str
    decision_reasons: list[str]
    text_score: float


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
