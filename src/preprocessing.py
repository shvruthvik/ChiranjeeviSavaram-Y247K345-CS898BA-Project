"""
Preprocessing pipeline for NEU-DET steel surface defect images.

Pipeline stages:
    1. Grayscale image loading
    2. CLAHE contrast enhancement
    3. Gaussian blur for noise suppression
    4. Canny edge extraction

This code file is used by:
    - generate_preprocessing_samples.py for qualitative before/after figures
    - baseline_classifier.py for classical baseline inputs
    - resnet50_transfer.py for multichannel ResNet50 inputs
"""

import cv2
import numpy as np


def load_grayscale(path):
    """Loads an image as a single-channel grayscale image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return img


def apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(gray)


def apply_gaussian_blur(gray, ksize=(5, 5), sigma=1.0):
    """Applies Gaussian smoothing to reduce small image noise."""
    return cv2.GaussianBlur(gray, ksize, sigma)


def apply_canny(gray, low_thresh=50, high_thresh=150):
    """Extracts image edges using the Canny edge detector."""
    return cv2.Canny(gray, low_thresh, high_thresh)


def full_pipeline(path):
    """
    Runs the complete preprocessing pipeline and returns every
    intermediate stage for visualization and analysis.
    """
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
    Produces the grayscale image used for intensity and texture features
    in the classical classifier.

    Processing:
        CLAHE -> Gaussian blur -> resize

    Canny edges are produced separately by
    preprocess_edges_for_classifier() so that edge features can be
    combined with intensity or texture features without discarding
    grayscale information.
    """
    raw = load_grayscale(path)
    clahe_img = apply_clahe(raw)
    blurred = apply_gaussian_blur(clahe_img)

    resized = cv2.resize(
        blurred,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    return resized


def preprocess_edges_for_classifier(path, target_size=(128, 128)):
    """
    Produces the Canny edge image used as an auxiliary feature source
    for the classical classifier.

    Processing:
        CLAHE -> Gaussian blur -> Canny -> resize
    """
    raw = load_grayscale(path)
    clahe_img = apply_clahe(raw)
    blurred = apply_gaussian_blur(clahe_img)
    edges = apply_canny(blurred)

    resized = cv2.resize(
        edges,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    return resized


def preprocess_multichannel_for_resnet(path, target_size=(224, 224)):
    """
    Creates a three-channel preprocessed representation using all image
    processing stages proposed in the project pitch.

    Channel 0:
        CLAHE-enhanced grayscale image

    Channel 1:
        CLAHE-enhanced image after Gaussian smoothing

    Channel 2:
        Canny edge map computed from the smoothed image

    This design preserves contrast, texture, and edge information while
    producing the three-channel input required by ResNet50.

    These are custom feature channels and do not represent RGB color
    channels.
    """
    raw = load_grayscale(path)
    clahe_img = apply_clahe(raw)
    blurred = apply_gaussian_blur(clahe_img)
    edges = apply_canny(blurred)

    ch0 = cv2.resize(
        clahe_img,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    ch1 = cv2.resize(
        blurred,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    ch2 = cv2.resize(
        edges,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    stacked = np.stack([ch0, ch1, ch2], axis=-1)

    return stacked