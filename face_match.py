"""
face_match.py
--------------
Wraps DeepFace to compare a face photo cropped from an ID document against
a live selfie, with sensible defaults and quality gating (blur, no-face-
detected) so the caller gets an actionable result rather than a raw
exception.

DeepFace is imported lazily inside functions rather than at module load
time - it pulls in TensorFlow and is slow to import, so this keeps
`import face_match` cheap for code paths (like tests) that don't need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2

from ocr_extraction import check_blur


@dataclass
class FaceMatchResult:
    matched: bool
    confidence: float          # 0-100, higher = more similar
    distance: float             # raw model distance (lower = more similar)
    threshold: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "confidence": round(self.confidence, 1),
            "distance": round(self.distance, 4),
            "threshold": self.threshold,
            "error": self.error,
        }


def _quality_check(image_path: str, blur_threshold: float = 80.0) -> Optional[str]:
    img = cv2.imread(image_path)
    if img is None:
        return "Could not read image file."
    is_blurry, score = check_blur(img, blur_threshold)
    if is_blurry:
        return f"Image is too blurry for reliable face matching (sharpness {score:.0f}, need >{blur_threshold:.0f})."
    return None


def match_faces(
    document_image_path: str,
    selfie_image_path: str,
    model_name: str = "Facenet512",
    distance_metric: str = "cosine",
    sensitivity: str = "balanced",
) -> FaceMatchResult:
    """
    Compares the face in the document image against the selfie.

    sensitivity: "strict" | "balanced" | "lenient" - maps to a tighter or
    looser distance threshold. Strict reduces false-accepts (good for
    high-stakes verification) at the cost of more false-rejects on poor
    lighting/angle; lenient does the reverse.
    """
    doc_quality_issue = _quality_check(document_image_path)
    selfie_quality_issue = _quality_check(selfie_image_path)
    if doc_quality_issue:
        return FaceMatchResult(False, 0.0, 1.0, 0.0, error=f"Document photo: {doc_quality_issue}")
    if selfie_quality_issue:
        return FaceMatchResult(False, 0.0, 1.0, 0.0, error=f"Selfie: {selfie_quality_issue}")

    try:
        from deepface import DeepFace
    except ImportError:
        return FaceMatchResult(
            False, 0.0, 1.0, 0.0,
            error="DeepFace is not installed. Run: pip install deepface tf-keras",
        )

    # DeepFace's default cosine threshold for Facenet512 is ~0.30; we scale
    # it per sensitivity level rather than hardcoding one strictness.
    threshold_scale = {"strict": 0.75, "balanced": 1.0, "lenient": 1.3}.get(sensitivity, 1.0)

    try:
        result = DeepFace.verify(
            img1_path=document_image_path,
            img2_path=selfie_image_path,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=True,
        )
    except ValueError as e:
        # DeepFace raises ValueError when it can't detect a face in one of
        # the two images - a very common real-world failure mode worth
        # surfacing distinctly from a genuine mismatch.
        return FaceMatchResult(False, 0.0, 1.0, 0.0, error=f"Face detection failed: {e}")
    except Exception as e:
        return FaceMatchResult(False, 0.0, 1.0, 0.0, error=f"Face matching error: {e}")

    base_threshold = result.get("threshold", 0.30)
    adjusted_threshold = base_threshold * threshold_scale
    distance = result.get("distance", 1.0)
    matched = distance <= adjusted_threshold

    # Convert distance to an intuitive 0-100 confidence score - not a
    # statistically calibrated probability, just a readable proxy where
    # 0 distance -> 100 confidence and distance-at-threshold -> ~65.
    confidence = max(0.0, min(100.0, 100 * (1 - distance / max(adjusted_threshold * 1.6, 1e-6))))

    return FaceMatchResult(
        matched=matched,
        confidence=confidence,
        distance=distance,
        threshold=adjusted_threshold,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python face_match.py <document_image> <selfie_image> [strict|balanced|lenient]")
        sys.exit(1)

    sensitivity = sys.argv[3] if len(sys.argv) > 3 else "balanced"
    res = match_faces(sys.argv[1], sys.argv[2], sensitivity=sensitivity)
    print(res.to_dict())
