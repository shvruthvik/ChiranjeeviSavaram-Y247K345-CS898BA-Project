# Steel Surface Defect Detection & Classification

### CS898BA – Image Analysis and Computer Vision

## Midterm Progress Report

**Student:** Chiranjeevi Venkata Shiva Ruthvik Savaram

---

# Project Overview

This repository contains my **Midterm Progress** submission for **CS898BA – Image Analysis and Computer Vision**. The project focuses on automated steel surface defect classification using both traditional machine learning and deep learning techniques. During this milestone, I implemented the image preprocessing pipeline, developed a classical machine learning baseline, built a transfer learning framework using ResNet50, and performed preliminary experiments to evaluate different approaches.

The project uses the **NEU Surface Defect Database (NEU-DET)**, which contains six categories of steel surface defects commonly used for benchmarking image classification algorithms.

---

# Project Objectives

The primary objectives of this project are:

- Develop a reusable image preprocessing pipeline.
- Build a traditional machine learning baseline.
- Implement a deep learning solution using transfer learning.
- Compare the performance of raw and preprocessed images.
- Establish a complete framework for the final project milestone.

---

# Dataset

**Dataset:** NEU Surface Defect Database (NEU-DET)

The dataset consists of grayscale steel surface images belonging to the following six defect classes:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

The provided training and validation split supplied with the dataset was used for all experiments. The complete dataset is included in this repository to allow the project to be reproduced without any additional downloads.

---

# Development Environment

- Python 3.13
- Windows 11
- Visual Studio Code
- Git
- GitHub

---

# Tasks Completed

## 1. Image Preprocessing

Implemented a reusable preprocessing pipeline consisting of:

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian Blur
- Canny Edge Detection

The preprocessing pipeline is used throughout the project to generate enhanced images for visualization and model training.

---

## 2. Traditional Machine Learning Baseline

Implemented a complete HOG + SVM classification pipeline.

The baseline includes:

- HOG feature extraction
- Feature fusion using HOG descriptors extracted from both the preprocessed grayscale image and the Canny edge image
- Support Vector Machine (SVM) classifier
- Performance evaluation using multiple classification metrics
- Comparison between raw and preprocessed image inputs

---

## 3. Transfer Learning

Implemented a ResNet50 transfer learning framework.

The implementation includes:

- Custom dataset loader
- Image preprocessing and normalization
- Transfer learning using an ImageNet-pretrained ResNet50 model
- Fine-tuning of the final residual block and classification layer
- Performance evaluation using both raw grayscale images and custom three-channel preprocessed images

---

## 4. Visualization

Generated several visualizations to summarize the preprocessing pipeline and model performance.

Generated figures include:

- Preprocessing pipeline comparison
- Baseline classifier metric comparison
- Confusion matrix comparison

---

# Midterm Experimental Results

| Model | Input | Accuracy |
|--------|-------|----------|
| HOG + SVM | Raw Images | **86.94%** |
| HOG + SVM | Preprocessed Feature Fusion | **89.44%** |
| ResNet50 | Raw Images | **99.72%** |
| ResNet50 | Preprocessed Images | **99.44%** |

---

# Observations

The experiments completed during this milestone produced several useful observations.

- The preprocessing pipeline improved the HOG + SVM baseline by approximately **2.5 percentage points** over the raw-image baseline.
- ResNet50 achieved very high classification accuracy for both input conditions.
- The raw grayscale representation slightly outperformed the custom multichannel preprocessing for ResNet50 in the current experiments.
- These results suggest that handcrafted preprocessing provides greater benefit to the traditional feature-based classifier than to the pretrained deep learning model used in this project.

---

# Repository Structure

```text
data/
    train/
    validation/

src/
    preprocessing.py
    baseline_classifier.py
    resnet50_transfer.py
    generate_preprocessing_samples.py
    generate_results_plots.py

outputs/
    baseline_results.json
    plots/
        preprocessing_pipeline_comparison.jpg
        baseline_metric_comparison.jpg
        confusion_matrices.jpg

README.md
AI_Log.md
requirements.txt
```

---

# Future Work

The remaining work for the final project includes:

- Hyperparameter tuning
- Learning-rate optimization
- Additional preprocessing experiments
- Evaluation using additional performance metrics
- Grad-CAM visualization for model explainability
- Final report and presentation

---

# Technologies Used

- Python
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- PyTorch
- Torchvision

---

# Conclusion

This midterm milestone establishes the core framework for the final project by implementing an image preprocessing pipeline, a traditional HOG + SVM baseline, and a ResNet50 transfer learning model for steel surface defect classification.

The experimental results show that the preprocessing pipeline improves the traditional machine learning approach, while the pretrained ResNet50 model performs slightly better using raw grayscale inputs. These findings provide a solid foundation for the remaining experiments, model improvements, and explainability techniques that will be completed during the final project milestone.