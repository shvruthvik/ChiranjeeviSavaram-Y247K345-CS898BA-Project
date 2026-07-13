# CS898BA Midterm Project
## Automated Surface Defect Classification in Manufacturing

### Student Information

**Name:** Chiranjeevi Venkata Shiva Ruthvik Savaram

---

# Project Overview

This repository contains the **Midterm Progress** submission for **CS898BA – Image Analysis and Computer Vision**. The project focuses on automated surface defect classification using both traditional machine learning and deep learning techniques. The work completed in this milestone establishes the preprocessing pipeline, baseline classifier, transfer learning framework, and preliminary experimental results that will be extended during the final project.

The project uses the **NEU Surface Defect Database (NEU-DET)**, which contains six categories of steel surface defects commonly used for image classification research.

---

# Objectives

- Develop an image preprocessing pipeline.
- Develop a classical machine learning baseline.
- Implement a deep learning solution using transfer learning.
- Compare the influence of preprocessing on both approaches.
- Prepare for the final project milestone.

---

# Dataset

**Dataset:** NEU Surface Defect Database (NEU-DET)

Classes:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

---

# Development Environment

- Python 3.x
- Windows 11
- Visual Studio Code
- Git & GitHub

---

# Tasks Completed

## 1. Image Preprocessing

Implemented a reusable preprocessing pipeline including:

- CLAHE
- Gaussian Blur
- Canny Edge Detection

The preprocessing pipeline produces enhanced images for later experimentation.

---

## 2. Traditional Machine Learning Baseline

Implemented a complete HOG + SVM classification pipeline.

Features:

- HOG feature extraction
- Support Vector Machine classifier
- Performance evaluation
- Comparison using raw and preprocessed images

---

## 3. Transfer Learning

Implemented a ResNet50 transfer learning pipeline.

Current implementation includes:

- Dataset loading
- Image transformations
- Transfer learning with ResNet50
- Model evaluation
- Classification metrics

---

## 4. Visualization

Generated visual outputs including:

- Preprocessing comparison figures
- Classification metric comparison charts

---

# Preliminary Experimental Results

| Model | Dataset | Accuracy |
|--------|---------|----------|
| HOG + SVM | Raw Images | 86.9% |
| HOG + SVM | Preprocessed Images | 89.4% |
| ResNet50 | Raw Images | 99.72% |
| ResNet50 | Preprocessed Images | 98.61% |

---

# Observations

- Image preprocessing improved the traditional HOG + SVM baseline.
- ResNet50 achieved excellent performance on the raw dataset.
- Preprocessing produced a slight decrease in the preliminary ResNet50 results.
- These observations will be investigated further during the final project milestone.

---

# Repository Structure

```
src/
    preprocessing.py
    baseline_classifier.py
    resnet50_transfer.py
    generate_preprocessing_samples.py
    generate_results_plots.py

outputs/
    plots/
    preprocessing_samples/
    baseline_results.json

README.md
AI_Log.md
requirements.txt
```

---

# Future Work

The remaining work includes:

- Hyperparameter tuning
- Additional experimentation
- Cross-validation
- Confusion matrix analysis
- Grad-CAM visualization
- Final report and presentation

---

# Technologies Used

- Python
- NumPy
- OpenCV
- scikit-learn
- PyTorch
- Torchvision
- Matplotlib

---

# Conclusion

This milestone establishes the core framework for the final project by implementing image preprocessing, a traditional machine learning baseline, and a transfer learning pipeline using ResNet50. Preliminary experiments demonstrate promising classification performance while highlighting how preprocessing affects different modeling approaches. Future work will focus on hyperparameter tuning, additional evaluation, explainability techniques, and a comprehensive comparison of all models.