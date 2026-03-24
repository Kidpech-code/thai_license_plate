from __future__ import annotations

from difflib import get_close_matches
from itertools import product
import re

from .geometry import preprocess_plate
from .types import DomainConfig


def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())


def build_plate_line_candidates(text_candidates):
    normalized_tokens = []
    seen_tokens = set()
    for candidate in text_candidates:
        cleaned = clean_text(candidate)
        if not cleaned:
            continue
        if not re.search(r"[0-9ก-ฮ]", cleaned):
            continue
        if cleaned in seen_tokens:
            continue
        seen_tokens.add(cleaned)
        normalized_tokens.append(cleaned)

    combined_candidates = list(normalized_tokens)
    for start_index in range(len(normalized_tokens)):
        for end_index in range(start_index + 2, min(len(normalized_tokens), start_index + 3) + 1):
            combined_candidates.append(" ".join(normalized_tokens[start_index:end_index]))

    deduplicated_candidates = []
    seen_candidates = set()
    for candidate in combined_candidates:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        deduplicated_candidates.append(candidate)
    return deduplicated_candidates


def extract_plate_letters(text):
    return "".join(char for char in text if "ก" <= char <= "ฮ")


def extract_plate_letters_before_digits(text, digits):
    if not text:
        return ""

    text_before_digits = text
    if digits:
        digit_match = re.search(re.escape(digits), text)
        if digit_match:
            text_before_digits = text[:digit_match.start()]

    return extract_plate_letters(text_before_digits)


def get_confusion_options(char, domain_config: DomainConfig):
    return domain_config.thai_plate_char_confusions.get(char, (char,))


def score_confusion_resolution(candidate, raw_letters, domain_config: DomainConfig):
    score = 0.0
    for raw_char, resolved_char in zip(raw_letters, candidate):
        options = get_confusion_options(raw_char, domain_config)
        if resolved_char not in options:
            continue

        if len(options) == 1:
            score += 1.8
            continue

        if resolved_char == raw_char:
            score += 0.6
            continue

        score += 0.3

    return score


def score_vehicle_type_prefix(candidate, vehicle_type, domain_config: DomainConfig):
    if not candidate:
        return 0.0

    exact_series = domain_config.valid_two_letter_series_by_vehicle_type.get(vehicle_type, ())
    all_known_series = {
        series
        for series_list in domain_config.valid_two_letter_series_by_vehicle_type.values()
        for series in series_list
    }
    if len(candidate) == 2 and exact_series and candidate in exact_series:
        series_rank = exact_series.index(candidate)
        return 6.0 + max(0.0, 1.5 - (0.1 * series_rank))
    if len(candidate) == 2 and candidate in all_known_series:
        return 2.5

    allowed_prefixes = domain_config.series_prefixes_by_vehicle_type[vehicle_type]
    all_known_prefixes = {
        prefix
        for prefixes in domain_config.series_prefixes_by_vehicle_type.values()
        for prefix in prefixes
    }
    if allowed_prefixes and candidate[0] in allowed_prefixes:
        prefix_rank = allowed_prefixes.index(candidate[0])
        return 3.0 + max(0.0, 1.5 - (0.25 * prefix_rank))
    if candidate[0] in all_known_prefixes:
        return 1.5
    return 0.0


def generate_plate_letter_candidates(raw_letters, domain_config: DomainConfig, max_candidates=32):
    if not raw_letters:
        return []

    expanded_candidates = []
    seen = set()

    raw_letter_variants = []
    for variant_length in (2, 3):
        if len(raw_letters) >= variant_length:
            raw_letter_variants.append(raw_letters[:variant_length])
    if raw_letters[:3] not in raw_letter_variants:
        raw_letter_variants.append(raw_letters[:3])

    for raw_letter_variant in raw_letter_variants:
        option_groups = [get_confusion_options(char, domain_config) for char in raw_letter_variant]
        for combination in product(*option_groups):
            candidate = "".join(combination)
            if candidate not in seen:
                seen.add(candidate)
                expanded_candidates.append(candidate)
            if len(expanded_candidates) >= max_candidates:
                break
        if len(expanded_candidates) >= max_candidates:
            break

    return expanded_candidates or [raw_letters[:3]]


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
    score += score_vehicle_type_prefix(candidate, vehicle_type, domain_config)

    return score


def normalize_plate_line(text_candidates, vehicle_type, domain_config: DomainConfig):
    best_candidate = ""
    best_score = -1.0

    for candidate in build_plate_line_candidates(text_candidates):
        cleaned = clean_text(candidate)
        digit_runs = re.findall(r"\d{1,4}", cleaned)
        digits = max(digit_runs, key=len) if digit_runs else ""
        prefix_digit = extract_series_prefix_digit(cleaned, digits)
        raw_letters = extract_plate_letters_before_digits(cleaned, digits)

        for letters in generate_plate_letter_candidates(raw_letters, domain_config):
            score = score_letter_candidate(letters, raw_letters, vehicle_type, domain_config)
            if digits:
                score += 2.0
                if len(digits) == 4:
                    score += 1.0
                elif len(digits) in (2, 3):
                    score += 0.5

                # The supported Thai formats in this pipeline are either
                # two letters plus serial digits, or prefix digit + two letters.
                if len(letters) == 2:
                    score += 2.0
                elif len(letters) == 3:
                    score -= 1.5

            if prefix_digit:
                score += 1.0
            if prefix_digit and len(letters) == 2 and len(digits) in (2, 3, 4):
                score += 2.5
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
    reasons = []
    has_strong_plate_pattern = False

    if re.fullmatch(r"[ก-ฮ]{2} \d{4}", plate_line):
        score += 4.0
        has_strong_plate_pattern = True
        reasons.append("plate_pattern_exact")
    elif re.fullmatch(r"\d[ก-ฮ]{2} \d{1,4}", plate_line):
        score += 4.0
        has_strong_plate_pattern = True
        reasons.append("plate_pattern_prefix_digit")
    elif re.search(r"\d{4}", plate_line):
        score += 2.0
        reasons.append("plate_has_four_digits")
    elif re.search(r"\d[ก-ฮ]{2} \d{1,4}", plate_line):
        score += 2.5
        reasons.append("plate_has_prefix_digit")
    elif plate_line:
        reasons.append("plate_partial")
    else:
        reasons.append("plate_missing")

    if province_line in domain_config.thai_provinces:
        score += 4.0
        reasons.append("province_exact")
    elif province_line:
        score += 1.0
        reasons.append("province_partial")
    else:
        reasons.append("province_missing")

    if province_line == "กรุงเทพมหานคร":
        score += 1.5
        reasons.append("province_bangkok_bonus")

    return score, reasons, has_strong_plate_pattern


def decide_result_status(total_score, has_strong_plate_pattern, province_line, domain_config: DomainConfig):
    if has_strong_plate_pattern and province_line in domain_config.thai_provinces and total_score >= domain_config.success_score_threshold:
        return "success"
    if total_score >= domain_config.low_confidence_score_threshold and (has_strong_plate_pattern or province_line):
        return "low_confidence"
    return "failed"
