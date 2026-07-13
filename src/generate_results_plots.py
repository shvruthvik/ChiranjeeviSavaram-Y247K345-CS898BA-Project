import json
import numpy as np
import matplotlib.pyplot as plt

CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

with open("../outputs/baseline_results.json") as f:
    results = json.load(f)

# --- Metric comparison bar chart ---
metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

raw = results[0]
prep = results[1]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
raw_vals = [raw[m] * 100 for m in metrics]
prep_vals = [prep[m] * 100 for m in metrics]

bars1 = ax.bar(x - width/2, raw_vals, width, label="Raw -> HOG -> SVM", color="#4C72B0")
bars2 = ax.bar(x + width/2, prep_vals, width, label="Preprocessed (CLAHE+Blur+Canny) -> HOG -> SVM", color="#DD8452")

ax.set_ylabel("Score (%)")
ax.set_title("Baseline Classifier: Raw vs. Preprocessed Input")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 100)
ax.legend()

for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("../outputs/plots/baseline_metric_comparison.jpg", dpi=150)
print("Saved: ../outputs/plots/baseline_metric_comparison.jpg")

# --- Confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
titles = ["Raw -> HOG -> SVM", "Preprocessed (CLAHE+Blur+Canny) -> HOG -> SVM"]

for ax, res, title in zip(axes, results, titles):
    cm = np.array(res["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)

plt.tight_layout()
plt.savefig("../outputs/plots/confusion_matrices.jpg", dpi=150)
print("Saved: ../outputs/plots/confusion_matrices.jpg")
