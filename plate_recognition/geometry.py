from __future__ import annotations

from pathlib import Path

import cv2


def load_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพหรือเปิดไฟล์ไม่ได้: {image_path}")
    return image


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
