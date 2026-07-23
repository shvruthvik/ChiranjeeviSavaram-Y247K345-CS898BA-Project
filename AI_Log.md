# AI Log

## Entry 1

**Date and Time:** July 11, 2026, 6:15 PM

**AI Tool:** Claude (Sonnet 5)

**Prompt:** I have to complete the Midterm Progress milestone for my CS898BA project. Can you help me understand the project requirements and suggest a plan for completing each part?

**Response Synopsis:** Received guidance on understanding the milestone requirements and breaking the work into manageable implementation tasks.

**Changes Made:** Planned the project workflow and organized the repository before beginning implementation.

---

## Entry 2

**Date and Time:** July 11, 2026, 6:45 PM

**AI Tool:** Claude (Sonnet 5)

**Prompt:** How should I design my preprocessing pipeline so I can reuse it throughout the project instead of writing the same code multiple times?

**Response Synopsis:** Received recommendations for creating reusable preprocessing functions and applying consistent image enhancement techniques.

**Changes Made:** Implemented reusable preprocessing functions for CLAHE, Gaussian blur, and Canny edge detection.

---

## Entry 3

**Date and Time:** July 11, 2026, 7:20 PM

**AI Tool:** Claude (Sonnet 5)

**Prompt:** What would be a good traditional machine learning baseline for this steel surface defect dataset?

**Response Synopsis:** Received guidance on using HOG features with an SVM classifier as the baseline model.

**Changes Made:** Implemented HOG feature extraction and trained the baseline SVM classifier.

---

## Entry 4

**Date and Time:** July 11, 2026, 8:00 PM

**AI Tool:** Claude (Sonnet 5)

**Prompt:** How can I fairly compare the performance of models trained using raw images and preprocessed images?

**Response Synopsis:** Received suggestions for evaluating both approaches using the same dataset split and evaluation metrics.

**Changes Made:** Generated baseline results for both image conditions and compared their performance.

---

## Entry 5

**Date and Time:** July 11, 2026, 8:30 PM

**AI Tool:** ChatGPT

**Prompt:** Can you review my preprocessing pipeline and check whether it matches the workflow I presented in my project proposal?

**Response Synopsis:** Reviewed the implementation and identified that the preprocessing pipeline did not fully match the proposed workflow because Canny edge information was not contributing to the classifier features.

**Changes Made:** Updated the preprocessing pipeline to incorporate Canny edge information into the feature extraction process and reran the baseline experiments.

---

## Entry 6

**Date and Time:** July 11, 2026, 9:00 PM

**AI Tool:** Claude (Sonnet 5)

**Prompt:** How should I implement transfer learning using ResNet50 for this image classification project?

**Response Synopsis:** Received guidance on dataset preparation, transfer learning configuration, and model evaluation.

**Changes Made:** Implemented the ResNet50 transfer learning pipeline.

---

## Entry 7

**Date and Time:** July 11, 2026, 9:45 PM

**AI Tool:** ChatGPT

**Prompt:** I'm getting errors while running my project. Can you help me identify what is causing them and how to fix them?

**Response Synopsis:** Helped troubleshoot execution issues, project paths, missing dependencies, and preprocessing pipeline inconsistencies.

**Changes Made:** Corrected project paths, resolved dependency issues, and verified successful execution of the preprocessing and baseline scripts.

---

## Entry 8

**Date and Time:** July 11, 2026, 10:20 PM

**AI Tool:** ChatGPT

**Prompt:** Can you help me generate figures that clearly show the preprocessing pipeline and compare my baseline model results?

**Response Synopsis:** Suggested creating qualitative preprocessing figures together with quantitative comparison charts and confusion matrices.

**Changes Made:** Generated preprocessing comparison figures, baseline metric comparison plots, and confusion matrix visualizations.

---

## Entry 9

**Date and Time:** July 12, 2026, 9:00 AM

**AI Tool:** ChatGPT

**Prompt:** Can you help me interpret my experimental results and explain why preprocessing affects the traditional model and ResNet50 differently?

**Response Synopsis:** Explained the observed differences between the HOG + SVM baseline and the ResNet50 model and helped summarize the experimental findings.

**Changes Made:** Documented the observations and updated the project results section.

---

## Entry 10

**Date and Time:** July 12, 2026, 10:00 AM

**AI Tool:** ChatGPT

**Prompt:** Can you review my README and help me make sure it matches my implementation, experimental results, and project requirements?

**Response Synopsis:** Reviewed the README for consistency with the implementation, corrected outdated experimental results, and improved the overall organization.

**Changes Made:** Updated the README with the final implementation details, verified experimental results, and improved the project documentation.

---

## Entry 11

**Date and Time:** July 12, 2026, 11:00 AM

**AI Tool:** ChatGPT

**Prompt:** Can you review my entire repository before submission and let me know if anything still needs to be fixed?

**Response Synopsis:** Reviewed the repository structure, source code, generated outputs, documentation, and project organization to ensure consistency before submission.

**Changes Made:** Finalized the repository, verified the generated outputs, cleaned unnecessary files, and prepared the project for submission.

---

## Entry 12

**Date and Time:** July 22, 2026, 6:30 PM

**AI Tool:** ChatGPT

**Prompt:** Can you help me design the hyperparameter tuning experiments required for the final project? I need to test learning rate, batch size, and fine-tuning strategy without making the experiment unnecessarily large.

**Response Synopsis:** Received guidance on selecting a focused set of ResNet50 configurations that changed one major training parameter at a time and allowed comparison between Layer4 + FC fine-tuning and classifier-only training.

**Changes Made:** Implemented the hyperparameter tuning script and evaluated five configurations using different learning rates, batch sizes, and fine-tuning strategies.

---

## Entry 13

**Date and Time:** July 22, 2026, 8:00 PM

**AI Tool:** ChatGPT

**Prompt:** Can you review the hyperparameter tuning results and help me decide which checkpoint should be used as the final model?

**Response Synopsis:** Reviewed the tuning results and compared the validation accuracy of each configuration. Three Layer4 + FC configurations achieved 100% validation accuracy, while the classifier-only configuration achieved 90%.

**Changes Made:** Selected the configuration using a learning rate of 0.0001, batch size of 32, and Layer4 + FC fine-tuning as the final model because it achieved the best performance with a simple and reproducible setup.

---

## Entry 14

**Date and Time:** July 22, 2026, 9:15 PM

**AI Tool:** ChatGPT

**Prompt:** Can you help me create a final evaluation script that loads the selected checkpoint and reports accuracy, precision, recall, F1-score, a classification report, and a confusion matrix?

**Response Synopsis:** Received guidance on rebuilding the ResNet50 architecture, loading the saved model checkpoint, running inference on the validation dataset, and calculating the required classification metrics.

**Changes Made:** Implemented `evaluate_model.py`, evaluated the final model on all 360 validation images, and saved the final metrics, classification report, and confusion matrix.

---

## Entry 15

**Date and Time:** July 22, 2026, 10:00 PM

**AI Tool:** ChatGPT

**Prompt:** My final model achieved 100% accuracy on the validation set. How should I report this result without making an unrealistic claim about perfect generalization?

**Response Synopsis:** Received guidance on distinguishing performance on the current validation set from generalization to unseen external data.

**Changes Made:** Added a limitation statement explaining that the perfect result applies to the specific 360-image validation set and should not be interpreted as proof of perfect performance on all future steel-defect images.

---

## Entry 16

**Date and Time:** July 23, 2026, 12:30 PM

**AI Tool:** ChatGPT

**Prompt:** Can you help me implement Grad-CAM for the final ResNet50 model and generate one visualization for each defect class?

**Response Synopsis:** Received guidance on registering hooks on the final convolutional layer, calculating class-specific gradients, producing activation heatmaps, and overlaying them on the original images.

**Changes Made:** Implemented the Grad-CAM visualization script and generated original-image, heatmap, and overlay comparisons for all six defect classes.

---

## Entry 17

**Date and Time:** July 23, 2026, 2:00 PM

**AI Tool:** ChatGPT

**Prompt:** Can you help me interpret the Grad-CAM results without overstating what the heatmaps prove?

**Response Synopsis:** Reviewed the Grad-CAM outputs and explained that localized defect classes showed clearer correspondence with visible defect regions, while diffuse texture-based classes produced broader activation patterns.

**Changes Made:** Updated the README to describe the Grad-CAM results as partial, class-dependent qualitative evidence rather than proof that every prediction was based only on the visible defect region.

---

## Entry 18

**Date and Time:** July 23, 2026, 4:00 PM

**AI Tool:** ChatGPT

**Prompt:** Can you review my virtual demonstration code and verify that the image loading and preprocessing match the ResNet50 training and evaluation pipeline?

**Response Synopsis:** Reviewed the demo implementation and identified that the demo should use the same grayscale OpenCV loading, 224 × 224 resizing, grayscale-to-RGB conversion, tensor conversion, and ImageNet normalization used during model evaluation.

**Changes Made:** Updated `demo_pipeline.py` so that its inference conditions match the final raw-image ResNet50 pipeline.

---

## Entry 19

**Date and Time:** July 23, 2026, 5:00 PM

**AI Tool:** ChatGPT

**Prompt:** Is selecting one validation image from each class enough for the required virtual demonstration, and what information should the output show?

**Response Synopsis:** Received guidance that a six-image batch demonstration was sufficient when combined with the complete validation evaluation. The recommended output included the actual class, predicted class, confidence score, and whether the prediction was correct.

**Changes Made:** Completed the batch-format virtual demonstration using one representative image from each defect class and saved the result as `outputs/demo/demo_predictions_raw.png`.

---

## Entry 20

**Date and Time:** July 23, 2026, 5:30 PM

**AI Tool:** ChatGPT

**Prompt:** Can you review the virtual demonstration results and help me document them in the README?

**Response Synopsis:** Reviewed the six predictions and recommended including the execution command, prediction table, confidence scores, saved output figure, and a short explanation of the results.

**Changes Made:** Added a Virtual Demonstration section to the README showing 6/6 correct predictions and confidence scores ranging from 93.33% to 99.95%.

---

## Entry 21

**Date and Time:** July 23, 2026, 7:00 PM

**AI Tool:** ChatGPT

**Prompt:** Can you review my final README and check whether it covers all required areas, including architecture, hyperparameter tuning, preprocessing analysis, metrics, Grad-CAM, and the virtual demonstration?

**Response Synopsis:** Reviewed the complete README and checked the consistency of the Table of Contents, pipeline diagram, experimental tables, observations, repository structure, execution commands, demonstration results, and conclusion.

**Changes Made:** Added the virtual demonstration to the Table of Contents, pipeline diagram, objectives, repository structure, running instructions, observations, and conclusion. Also corrected formatting and improved cautious wording around model generalization and Grad-CAM interpretation.

---

## Entry 22

**Date and Time:** July 23, 2026, 8:00 PM

**AI Tool:** ChatGPT

**Prompt:** Can you help me identify unnecessary or duplicate output files before I finalize the repository?

**Response Synopsis:** Reviewed the generated checkpoints and result files and identified duplicate or outdated outputs that were no longer needed for the final project.

**Changes Made:** Removed unnecessary duplicate files while retaining the final trained checkpoint, tuning results, evaluation metrics, classification report, plots, Grad-CAM outputs, and demonstration image.

---

## Entry 23

**Date and Time:** July 23, 2026, 9:00 PM

**AI Tool:** ChatGPT

**Prompt:** Can you perform one final consistency review of the project documentation and confirm whether the README and AI log match the actual implementation?

**Response Synopsis:** Reviewed the final project workflow from preprocessing through virtual demonstration and checked that the documented scripts, numerical results, generated outputs, and repository paths were consistent.

**Changes Made:** Performed final documentation corrections and prepared the README, AI log, source code, and output folders for submission.