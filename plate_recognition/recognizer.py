from __future__ import annotations

import logging
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLOWorld

from .geometry import clamp_box, expand_box, is_plate_like, load_image
from .normalization import decide_result_status, read_plate_lines, score_plate_result
from .types import AppConfig, DetectionCandidate, DomainConfig, PlateResult


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

    def collect_yolo_detections_for_prompts(self, image, prompts):
        image_height, image_width = image.shape[:2]
        detections = []
        for prompt in prompts:
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

    def collect_yolo_detections(self, image, contour_candidates):
        cumulative = []
        for batch_index, prompt_batch in enumerate(self.domain_config.prompt_batches, start=1):
            logging.debug("รัน YOLO-World prompt batch %s: %s", batch_index, ", ".join(prompt_batch))
            cumulative.extend(self.collect_yolo_detections_for_prompts(image, prompt_batch))
            merged = self.merge_candidates(cumulative + contour_candidates)
            strong_yolo_candidates = [
                candidate
                for candidate in merged
                if candidate.source == "yolo" and candidate.confidence >= self.domain_config.yolo_early_acceptance_confidence
            ]
            if strong_yolo_candidates:
                logging.debug(
                    "หยุด prompt cascade ที่ batch %s เพราะพบ strong yolo candidate %.4f",
                    batch_index,
                    strong_yolo_candidates[0].confidence,
                )
                break
        return cumulative

    def process_image(self, image_path: Path):
        logging.info("กำลังอ่านภาพ: %s", image_path)
        image = load_image(image_path)
        contour_candidates = self.find_plate_by_contours(image)
        logging.info("กำลังค้นหาป้ายทะเบียนด้วย prompt cascade %s batches", len(self.domain_config.prompt_batches))
        yolo_candidates = self.collect_yolo_detections(image, contour_candidates)
        merged_candidates = self.merge_candidates(contour_candidates + yolo_candidates)

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
            text_score, decision_reasons, has_strong_plate_pattern = score_plate_result(
                plate_line,
                province_line,
                self.domain_config,
            )
            score = text_score + candidate.confidence
            decision_status = decide_result_status(
                score,
                has_strong_plate_pattern,
                province_line,
                self.domain_config,
            )
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
                    decision_status=decision_status,
                    decision_reasons=decision_reasons,
                    text_score=text_score,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return image, merged_candidates, results, results[0]
