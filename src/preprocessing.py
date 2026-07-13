"""
Preprocessing pipeline for NEU-DET steel surface defect images.

Pipeline stages:
    1. Raw grayscale input (200x200)
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Gaussian blur (noise suppression)
    4. Canny edge detection (feature emphasis)

This module is used by both:
    - generate_preprocessing_samples() -> qualitative before/after figure for the README
    - extract_and_cache_features.py -> feeds preprocessed images into the classical baseline
"""

import cv2
import numpy as np


def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def apply_gaussian_blur(gray, ksize=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(gray, ksize, sigma)


def apply_canny(gray, low_thresh=50, high_thresh=150):
    return cv2.Canny(gray, low_thresh, high_thresh)


def full_pipeline(path):
    """Returns dict of every intermediate stage for a single image path."""
    raw = load_grayscale(path)
    clahe_img = apply_clahe(raw)
    blurred = apply_gaussian_blur(clahe_img)
    edges = apply_canny(blurred)
    return {
        "raw": raw,
        "clahe": clahe_img,
        "blurred": blurred,
        "edges": edges,
    }


def preprocess_for_classifier(path, target_size=(128, 128)):
    """
    Produces the final preprocessed image used as classifier input:
    CLAHE -> Gaussian blur -> resize.
    (Canny is generated separately as an auxiliary edge-feature channel,
    since feeding pure edge maps alone discards texture information that
    matters for defect classes like 'patches' and 'rolled-in_scale'.)
    """
    raw = load_grayscale(path)
    clahe_img = apply_clahe(raw)
    blurred = apply_gaussian_blur(clahe_img)
    resized = cv2.resize(blurred, target_size, interpolation=cv2.INTER_AREA)
    return resized


def preprocess_edges_for_classifier(path, target_size=(128, 128)):
    raw = load_grayscale(path)
    clahe_img = apply_clahe(raw)
    blurred = apply_gaussian_blur(clahe_img)
    edges = apply_canny(blurred)
    resized = cv2.resize(edges, target_size, interpolation=cv2.INTER_AREA)
    return resized


def preprocess_multichannel_for_resnet(path, target_size=(224, 224)):
    """
    Full-pipeline preprocessed input matching the pitch deck's stage order
    exactly: Raw -> CLAHE -> Gaussian Blur -> Canny Edge -> (fed to ResNet50).

    ResNet50 requires 3-channel input, so rather than discarding texture by
    feeding only the single-channel edge map (Canny alone loses the
    grayscale intensity information ResNet50's pretrained filters expect),
    every pipeline stage after the raw input is encoded as its own channel:
        channel 0: CLAHE-enhanced grayscale
        channel 1: CLAHE + Gaussian-blurred grayscale
        channel 2: CLAHE + Gaussian blur + Canny edge map
    This keeps all three preprocessing stages from the deck in the model's
    input exactly as pitched, while still preserving intensity/texture
    information alongside the edge geometry.
    """
    raw = load_grayscale(path)
    clahe_img = apply_clahe(raw)
    blurred = apply_gaussian_blur(clahe_img)
    edges = apply_canny(blurred)

    ch0 = cv2.resize(clahe_img, target_size, interpolation=cv2.INTER_AREA)
    ch1 = cv2.resize(blurred, target_size, interpolation=cv2.INTER_AREA)
    ch2 = cv2.resize(edges, target_size, interpolation=cv2.INTER_AREA)

    stacked = np.stack([ch0, ch1, ch2], axis=-1)
    return stacked
