from __future__ import annotations

import logging
import sys

from .config import build_output_paths, iter_input_images, load_domain_config, parse_args, setup_logging
from .evaluation import evaluate_predictions, load_ground_truth, write_evaluation_reports
from .recognizer import PlateRecognizer
from .reporting import annotate_and_save, save_crop, write_debug_json, write_result_json, write_summary_files
from .types import AppConfig


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
            logging.info("Decision status: %s", best_result.decision_status)
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
                    "status": best_result.decision_status,
                    "vehicle_type": best_result.vehicle_type,
                    "plate_text": best_result.plate_line,
                    "province": best_result.province_line,
                    "combined_text": f"{best_result.plate_line}\n{best_result.province_line}".strip(),
                    "source": best_result.source,
                    "prompt": best_result.prompt,
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
                    "status": "failed",
                    "vehicle_type": "",
                    "plate_text": "",
                    "province": "",
                    "combined_text": "",
                    "source": "",
                    "prompt": "",
                    "confidence": "",
                    "score": "",
                    "text_score": "",
                    "decision_reasons": [str(exc)],
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
        logging.info(
            "Evaluation summary: success_rate=%.4f low_confidence_rate=%.4f plate_exact_accuracy=%.4f province_exact_accuracy=%.4f combined_exact_accuracy=%.4f mean_combined_cer=%.4f",
            aggregate["success_rate"],
            aggregate["low_confidence_rate"],
            aggregate["plate_exact_accuracy"],
            aggregate["province_exact_accuracy"],
            aggregate["combined_exact_accuracy"],
            aggregate["mean_combined_cer"],
        )
        logging.info(
            "Detection summary: mean_iou=%.4f detection_iou_at_0_5=%.4f",
            aggregate["mean_iou"],
            aggregate["detection_iou_at_0_5"],
        )
        logging.info("บันทึก evaluation JSON ไว้ที่: %s", report_paths["evaluation_json"])
        logging.info("บันทึก evaluation CSV ไว้ที่: %s", report_paths["evaluation_csv"])
        logging.info("บันทึก leaderboard JSON ไว้ที่: %s", report_paths["leaderboard_json"])
        logging.info("บันทึก leaderboard CSV ไว้ที่: %s", report_paths["leaderboard_csv"])
        logging.info("บันทึก Thai confusion JSON ไว้ที่: %s", report_paths["confusion_json"])
        logging.info("บันทึก Thai confusion CSV ไว้ที่: %s", report_paths["confusion_csv"])
    sys.exit(exit_code)
