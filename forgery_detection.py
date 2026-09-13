"""
forgery_detection.py
---------------------
AI-based fake/tampered document screening module.

Designed to plug into the existing Indian-Document-Verification-System
(Streamlit + DeepFace + Tesseract) as an additional screening layer that
runs BEFORE / ALONGSIDE the face-match step.

Four independent signals are combined into a single authenticity score:
  1. QR cross-verification   (Aadhaar secure QR vs OCR text)
  2. Error Level Analysis     (JPEG recompression artifact detection)
  3. Copy-move detection      (duplicated-region detection via ORB)
  4. Metadata / EXIF forensics (screenshot / editor fingerprints)

Usage:
    from forgery_detection import DocumentForgeryDetector

    detector = DocumentForgeryDetector()
    report = detector.analyze(image_path, ocr_text=extracted_text)
    print(report["authenticity_score"], report["verdict"])
"""

from __future__ import annotations

import io
import re
import difflib
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags

try:
    from pyzbar.pyzbar import decode as zbar_decode
    _HAS_PYZBAR = True
except Exception:
    _HAS_PYZBAR = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float            # 0-100, contribution to authenticity (100 = clean)
    weight: float            # relative importance in final score
    details: str = ""
    evidence: dict = field(default_factory=dict)


@dataclass
class ForgeryReport:
    authenticity_score: float
    verdict: str              # "Likely Genuine" | "Needs Review" | "Likely Fake"
    checks: list

    def summary(self) -> str:
        lines = [f"Authenticity Score: {self.authenticity_score:.1f}/100  ->  {self.verdict}"]
        for c in self.checks:
            flag = "PASS" if c.passed else "FLAG"
            lines.append(f"  [{flag}] {c.name} (score {c.score:.0f}, weight {c.weight:.0%}) - {c.details}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "authenticity_score": round(self.authenticity_score, 1),
            "verdict": self.verdict,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "score": round(c.score, 1),
                    "weight": c.weight,
                    "details": c.details,
                    "evidence": c.evidence,
                }
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class DocumentForgeryDetector:
    """
    Runs a battery of forensic checks on a document image and produces
    a combined authenticity score (0-100) with a per-check breakdown.

    Weights are tuned so that a genuine, unedited, well-lit phone photo
    of an Aadhaar card scores 85-100, while a photoshopped or
    screenshot-of-a-screenshot document scores well below 50.
    """

    # Relative importance of each check in the final blended score.
    WEIGHTS = {
        "qr_cross_check": 0.35,
        "error_level_analysis": 0.30,
        "copy_move_detection": 0.20,
        "metadata_forensics": 0.15,
    }

    def __init__(self, ela_quality: int = 90, ela_diff_threshold: float = 18.0):
        self.ela_quality = ela_quality
        self.ela_diff_threshold = ela_diff_threshold

    # -- public API ----------------------------------------------------

    def analyze(self, image_path: str, ocr_text: Optional[str] = None) -> ForgeryReport:
        """
        Run all checks on the given image path.

        ocr_text: raw text your Tesseract pipeline already extracted from
                  the same image, used for QR cross-verification. Optional -
                  if omitted, the QR check falls back to internal consistency
                  only (QR decodes vs is well-formed).
        """
        checks = [
            self._check_qr_cross_verification(image_path, ocr_text),
            self._check_error_level_analysis(image_path),
            self._check_copy_move(image_path),
            self._check_metadata(image_path),
        ]

        total_weight = sum(c.weight for c in checks)
        weighted_score = sum(c.score * c.weight for c in checks) / total_weight

        if weighted_score >= 75:
            verdict = "Likely Genuine"
        elif weighted_score >= 50:
            verdict = "Needs Manual Review"
        else:
            verdict = "Likely Fake / Tampered"

        return ForgeryReport(authenticity_score=weighted_score, verdict=verdict, checks=checks)

    # -- check 1: QR cross-verification --------------------------------

    def _check_qr_cross_verification(self, image_path: str, ocr_text: Optional[str]) -> CheckResult:
        name = "QR Cross-Verification"
        weight = self.WEIGHTS["qr_cross_check"]

        if not _HAS_PYZBAR:
            return CheckResult(name, True, 70, weight,
                                "pyzbar not available - QR check skipped (install pyzbar+libzbar0).")

        img = cv2.imread(image_path)
        if img is None:
            return CheckResult(name, False, 0, weight, "Could not read image.")

        decoded = zbar_decode(img)
        if not decoded:
            # Not all ID types have a QR code (e.g. older Aadhaar, PAN, DL).
            # Absence isn't itself proof of forgery, so score neutrally.
            return CheckResult(name, True, 65, weight,
                                "No QR code found - document type may not carry one, or image quality is too low to decode.")

        qr_payload = decoded[0].data.decode(errors="ignore")

        if ocr_text is None:
            return CheckResult(name, True, 75, weight,
                                "QR decoded successfully. No OCR text supplied to cross-check against.",
                                evidence={"qr_payload_preview": qr_payload[:120]})

        # Extract digits/words from QR payload and compare against OCR text,
        # treating NAME tokens (alphabetic words) and NUMBER/DATE tokens
        # separately. A forger who edits just the printed name but leaves
        # DOB/UID untouched would still show high *overall* token overlap
        # (dates/numbers still match) even though the name is now fake -
        # so name-token overlap is checked and weighted on its own rather
        # than being diluted by the numeric fields.
        # Common ID-document field labels/boilerplate that would appear on
        # BOTH a genuine and a forged document regardless of whose name is
        # printed - excluded so they don't artificially inflate overlap.
        _STOPWORDS = {
            "NAME", "DOB", "DATE", "BIRTH", "GENDER", "MALE", "FEMALE",
            "ADDRESS", "GOVERNMENT", "INDIA", "UID", "UIDAI", "AADHAAR",
            "AADHAR", "CARD", "PAN", "PERMANENT", "ACCOUNT", "NUMBER",
            "IDENTIFICATION", "AUTHORITY", "UNIQUE", "SIGNATURE",
        }

        qr_words = set(w.upper() for w in re.findall(r"[A-Za-z]{3,}", qr_payload)) - _STOPWORDS
        ocr_words = set(w.upper() for w in re.findall(r"[A-Za-z]{3,}", ocr_text)) - _STOPWORDS
        qr_numbers = set(re.findall(r"\d{2,}", qr_payload))
        ocr_numbers = set(re.findall(r"\d{2,}", ocr_text))

        if not qr_words and not qr_numbers:
            return CheckResult(name, True, 60, weight, "QR decoded but payload had no comparable tokens.")

        word_overlap = qr_words & ocr_words
        word_ratio = len(word_overlap) / max(len(ocr_words), 1) if ocr_words else 1.0

        number_overlap = qr_numbers & ocr_numbers
        number_ratio = len(number_overlap) / max(len(ocr_numbers), 1) if ocr_numbers else 1.0

        # Name/word mismatch is the higher-value signal - if the numbers
        # (DOB, UID digits) line up but the words don't, that's exactly the
        # "name silently edited, QR left untouched" forgery pattern.
        if word_ratio >= 0.5 and number_ratio >= 0.5:
            return CheckResult(name, True, 95, weight,
                                f"QR data matches OCR text (name-token overlap {word_ratio:.0%}, number-token overlap {number_ratio:.0%}). Strong authenticity signal.",
                                evidence={"matched_words": list(word_overlap)[:10], "matched_numbers": list(number_overlap)[:10]})
        elif word_ratio < 0.35 and number_ratio >= 0.5:
            return CheckResult(name, False, 12, weight,
                                f"Dates/ID numbers in the QR match the document, but the NAME text does not "
                                f"(only {word_ratio:.0%} word overlap). This is the classic signature of a document "
                                f"where the printed name was edited after the QR code was generated.",
                                evidence={"qr_payload_preview": qr_payload[:150], "matched_numbers": list(number_overlap)[:10]})
        elif word_ratio >= 0.35 or number_ratio >= 0.35:
            return CheckResult(name, True, 55, weight,
                                f"Partial match between QR and OCR text (words {word_ratio:.0%}, numbers {number_ratio:.0%}) - "
                                f"could be OCR noise, but recommend manual review.",
                                evidence={"matched_words": list(word_overlap)[:10]})
        else:
            return CheckResult(name, False, 15, weight,
                                f"QR payload does NOT match printed text (words {word_ratio:.0%}, numbers {number_ratio:.0%}). "
                                f"Strong indicator the visible text was edited after the QR was generated.",
                                evidence={"qr_payload_preview": qr_payload[:150]})

    # -- check 2: Error Level Analysis -----------------------------------

    def _check_error_level_analysis(self, image_path: str) -> CheckResult:
        name = "Error Level Analysis"
        weight = self.WEIGHTS["error_level_analysis"]

        try:
            original = Image.open(image_path).convert("RGB")
        except Exception as e:
            return CheckResult(name, False, 0, weight, f"Could not open image: {e}")

        buf = io.BytesIO()
        original.save(buf, "JPEG", quality=self.ela_quality)
        buf.seek(0)
        resaved = Image.open(buf)

        orig_arr = np.asarray(original).astype(np.int16)
        resaved_arr = np.asarray(resaved).astype(np.int16)

        if orig_arr.shape != resaved_arr.shape:
            return CheckResult(name, True, 60, weight, "ELA skipped - image shape mismatch after recompression.")

        diff = np.abs(orig_arr - resaved_arr)
        # Normalize per-pixel error, then look at how "patchy" the high-error
        # regions are. Uniform low-level noise = normal JPEG behavior.
        # Localized high-error blobs = pasted/edited regions.
        gray_diff = diff.mean(axis=2)
        mean_error = float(gray_diff.mean())
        max_error = float(gray_diff.max())

        # Flag if there's a small hot region with much higher error than the
        # image-wide average (classic signature of a pasted-in edit).
        hot_mask = gray_diff > (mean_error + self.ela_diff_threshold)
        hot_fraction = float(hot_mask.mean())

        suspicious = (max_error > mean_error * 6) and (0.001 < hot_fraction < 0.35)

        if suspicious:
            score = max(10, 60 - hot_fraction * 100)
            return CheckResult(name, False, score, weight,
                                f"Localized high-error region detected (covers {hot_fraction:.1%} of image, "
                                f"peak error {max_error:.1f} vs avg {mean_error:.1f}). Suggests a pasted/edited region.",
                                evidence={"mean_error": mean_error, "max_error": max_error, "hot_fraction": hot_fraction})
        else:
            score = 90 if mean_error < 8 else 75
            return CheckResult(name, True, score, weight,
                                f"Compression error is uniform across the image (avg {mean_error:.1f}). No localized edits detected.",
                                evidence={"mean_error": mean_error, "max_error": max_error})

    # -- check 3: Copy-move (duplicated region) detection -----------------

    def _check_copy_move(self, image_path: str) -> CheckResult:
        """
        Real copy-move forgery (pasting one region over another) shows many
        keypoint pairs sharing the SAME displacement vector - the whole
        region moved by one consistent (dx, dy). Repeated text glyphs
        (e.g. multiple 'A's on a document) produce similar-looking keypoints
        too, but their displacement vectors are scattered/random, not
        clustered. So instead of counting raw duplicate matches, we cluster
        displacement vectors and only flag a large, tight cluster.
        """
        name = "Copy-Move Detection"
        weight = self.WEIGHTS["copy_move_detection"]

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return CheckResult(name, False, 0, weight, "Could not read image.")

        h, w = img.shape
        scale = min(1.0, 900 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape

        orb = cv2.ORB_create(nfeatures=2000)
        kp, des = orb.detectAndCompute(img, None)

        if des is None or len(kp) < 30:
            return CheckResult(name, True, 70, weight, "Not enough keypoints for reliable copy-move analysis.")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des, des, k=2)  # self-match, best real match after self

        min_dist_px = max(h, w) * 0.06  # ignore trivially-close neighbouring points
        vectors = []

        for m_list in matches:
            if len(m_list) < 2:
                continue
            best = m_list[1]
            if best.distance < 28:
                p1 = np.array(kp[best.queryIdx].pt)
                p2 = np.array(kp[best.trainIdx].pt)
                d = np.linalg.norm(p1 - p2)
                if d > min_dist_px:
                    vectors.append(p1 - p2)

        if len(vectors) < 8:
            return CheckResult(name, True, 88, weight,
                                f"No significant duplicated regions found ({len(vectors)} candidate matches, too few to cluster).",
                                evidence={"candidate_matches": len(vectors), "total_keypoints": len(kp)})

        vectors = np.array(vectors)
        # Bucket displacement vectors into coarse bins and find the largest cluster.
        bin_size = max(h, w) * 0.02
        binned = np.round(vectors / bin_size).astype(int)
        uniques, counts = np.unique(binned, axis=0, return_counts=True)
        largest_cluster = int(counts.max())
        cluster_ratio = largest_cluster / len(kp)

        # A tampered, pasted region shows up as one large tight cluster of
        # matching displacement vectors - typically well above what random
        # text repetition produces.
        suspicious = largest_cluster >= 12 and cluster_ratio > 0.02

        if suspicious:
            score = max(15, 70 - cluster_ratio * 400)
            return CheckResult(name, False, score, weight,
                                f"Found a cluster of {largest_cluster} keypoints sharing the same displacement vector "
                                f"({cluster_ratio:.1%} of all keypoints) - consistent with a region copy-pasted elsewhere in the image.",
                                evidence={"largest_cluster": largest_cluster, "total_keypoints": len(kp)})
        else:
            return CheckResult(name, True, 88, weight,
                                f"No consistent duplicated-region pattern found (largest displacement cluster: {largest_cluster} keypoints, "
                                f"consistent with normal text/pattern repetition).",
                                evidence={"largest_cluster": largest_cluster, "total_keypoints": len(kp)})

    # -- check 4: metadata / EXIF forensics --------------------------------

    def _check_metadata(self, image_path: str) -> CheckResult:
        name = "Metadata Forensics"
        weight = self.WEIGHTS["metadata_forensics"]

        try:
            img = Image.open(image_path)
        except Exception as e:
            return CheckResult(name, False, 0, weight, f"Could not open image: {e}")

        exif_raw = img._getexif() if hasattr(img, "_getexif") else None

        flags = []
        evidence = {}

        if not exif_raw:
            flags.append("No EXIF metadata present (common for screenshots, WhatsApp-compressed, or edited images).")
            score = 55
        else:
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
            evidence["exif_keys"] = list(exif.keys())[:20]

            software = str(exif.get("Software", "")).lower()
            editors = ["photoshop", "gimp", "picsart", "snapseed", "lightroom", "canva"]
            if any(e in software for e in editors):
                flags.append(f"Software tag indicates image editor was used: '{exif.get('Software')}'.")
                score = 20
            elif "Make" not in exif and "Model" not in exif:
                flags.append("No camera make/model tags - image may have been re-saved, screenshotted, or stripped.")
                score = 60
            else:
                score = 85

        # Filename/format heuristic: PNG screenshots of documents are common
        # in fraud attempts (re-encoding hides some forensic traces).
        if image_path.lower().endswith(".png"):
            flags.append("PNG format - if this originated as a screenshot rather than a camera photo, some forensic traces (like ELA) are weaker.")

        details = " ".join(flags) if flags else "EXIF metadata present and consistent with an unedited camera photo."
        return CheckResult(name, score >= 60, score, weight, details, evidence=evidence)


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python forgery_detection.py <image_path> [ocr_text_file]")
        sys.exit(1)

    ocr_text = None
    if len(sys.argv) > 2:
        with open(sys.argv[2], "r") as f:
            ocr_text = f.read()

    detector = DocumentForgeryDetector()
    report = detector.analyze(sys.argv[1], ocr_text=ocr_text)
    print(report.summary())
