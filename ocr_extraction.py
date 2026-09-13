"""
ocr_extraction.py
-------------------
Extracts text and structured fields (name, DOB, gender, ID number) from
Indian identity document images using Tesseract OCR, plus basic image
quality checks (blur detection) that gate whether OCR is even attempted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@dataclass
class OCRResult:
    raw_text: str
    name: Optional[str] = None
    dob: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    id_number: Optional[str] = None
    is_blurry: bool = False
    sharpness_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# Regex patterns tuned for common Indian ID layouts. These are intentionally
# permissive (OCR output is noisy) and only used as best-effort extraction -
# never as the sole trust signal (that's what forgery_detection.py's QR
# cross-check is for).
_DOB_PATTERNS = [
    r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b",           # 15/08/1995 or 15-08-1995
    r"\b(\d{4}[/-]\d{2}[/-]\d{2})\b",           # 1995-08-15
]
_AADHAAR_PATTERN = r"\b(\d{4}\s?\d{4}\s?\d{4})\b"
_PAN_PATTERN = r"\b([A-Z]{5}\d{4}[A-Z])\b"
_GENDER_PATTERN = r"\b(MALE|FEMALE|TRANSGENDER)\b"
_NAME_LABEL_PATTERN = r"(?:Name|नाम)\s*[:\-]?\s*([A-Za-z\s]{3,40}?)(?:\n|$)"


def check_blur(image_bgr: np.ndarray, threshold: float = 100.0) -> tuple[bool, float]:
    """
    Variance-of-Laplacian blur detection - a sharp image has high-frequency
    edges producing high variance; a blurry one is smoothed out and scores
    low. This gates OCR/face-match attempts on unusable images rather than
    silently returning garbage extractions.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, float(variance)


def _preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + adaptive threshold + slight denoise tends to noticeably
    improve Tesseract accuracy on phone-camera photos of ID cards compared
    to feeding it the raw color image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return thresh


def _calculate_age(dob_str: str) -> Optional[int]:
    for fmt_sep in ("/", "-"):
        parts = dob_str.replace("/", "-").split("-")
        if len(parts) != 3:
            continue
        try:
            if len(parts[0]) == 4:  # YYYY-MM-DD
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            else:  # DD-MM-YYYY
                d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
            born = date(y, m, d)
            today = date.today()
            age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
            if 0 < age < 130:
                return age
        except (ValueError, IndexError):
            continue
    return None


def extract_fields(image_path: str, blur_threshold: float = 100.0) -> OCRResult:
    img = cv2.imread(image_path)
    if img is None:
        return OCRResult(raw_text="", is_blurry=True, sharpness_score=0.0)

    is_blurry, sharpness = check_blur(img, blur_threshold)

    processed = _preprocess_for_ocr(img)
    raw_text = pytesseract.image_to_string(processed)
    # Fall back to the color image if preprocessing hurt more than it helped
    # (thin/ornate fonts on some ID card backgrounds can be over-thresholded).
    if len(raw_text.strip()) < 10:
        raw_text = pytesseract.image_to_string(img)

    result = OCRResult(raw_text=raw_text, is_blurry=is_blurry, sharpness_score=sharpness)

    dob_match = None
    for pattern in _DOB_PATTERNS:
        m = re.search(pattern, raw_text)
        if m:
            dob_match = m.group(1)
            break
    if dob_match:
        result.dob = dob_match
        result.age = _calculate_age(dob_match)

    gender_match = re.search(_GENDER_PATTERN, raw_text, re.IGNORECASE)
    if gender_match:
        result.gender = gender_match.group(1).upper()

    aadhaar_match = re.search(_AADHAAR_PATTERN, raw_text)
    pan_match = re.search(_PAN_PATTERN, raw_text)
    if aadhaar_match:
        result.id_number = aadhaar_match.group(1).replace(" ", "")
    elif pan_match:
        result.id_number = pan_match.group(1)

    name_match = re.search(_NAME_LABEL_PATTERN, raw_text, re.IGNORECASE)
    if name_match:
        candidate = name_match.group(1).strip()
        # Trim trailing OCR noise - keep to first 4 words, which covers
        # virtually all Indian name formats printed on ID cards.
        result.name = " ".join(candidate.split()[:4])

    return result


def detect_document_type(raw_text: str) -> str:
    text_upper = raw_text.upper()
    if re.search(_AADHAAR_PATTERN, raw_text) or "AADHAAR" in text_upper or "UIDAI" in text_upper:
        return "aadhaar"
    if re.search(_PAN_PATTERN, raw_text) or "INCOME TAX" in text_upper or "PERMANENT ACCOUNT" in text_upper:
        return "pan"
    if "DRIVING LICENCE" in text_upper or "DRIVING LICENSE" in text_upper or "TRANSPORT" in text_upper:
        return "driving_license"
    if "PASSPORT" in text_upper:
        return "passport"
    return "unknown"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_extraction.py <image_path>")
        sys.exit(1)

    res = extract_fields(sys.argv[1])
    print("Raw text:\n", res.raw_text[:300])
    print("\nExtracted fields:")
    for k, v in res.to_dict().items():
        if k != "raw_text":
            print(f"  {k}: {v}")
    print(f"\nDocument type guess: {detect_document_type(res.raw_text)}")
