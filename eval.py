"""Evaluation metrics and reporting for topic/cluster predictions."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
plt.rcParams["font.family"] = "Segoe UI Emoji"


def evaluate(y_true, y_pred):
    """Return accuracy and macro-averaged F1 as a dict of floats."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def confusion(y_true, y_pred):
    """Return the confusion matrix (rows = true, columns = pred)."""
    return confusion_matrix(y_true, y_pred)


def top_confusions(cm, target_names=None, top_n=10):
    """Return the top off-diagonal confusion pairs (most often confused classes)."""
    cm2 = cm.copy().astype(int)
    # Zero diagonal so we only rank misclassifications
    np.fill_diagonal(cm2, 0)

    pairs = []
    rows, cols = cm2.shape
    for i in range(rows):
        for j in range(cols):
            if cm2[i, j] > 0:
                pairs.append((cm2[i, j], i, j))

    # Sort by count descending
    pairs.sort(reverse=True, key=lambda x: x[0])

    out = []
    for count, i, j in pairs[:top_n]:
        # Handle case where target_names might be None
        name_i = target_names[i] if target_names else str(i)
        name_j = target_names[j] if target_names else str(j)
        out.append({"count": int(count), "true": name_i, "pred": name_j})

    return out


def save_confusion_plot(y_true, y_pred, target_names, filepath, metrics):
    """
    Generates a high-contrast confusion matrix.
    - Hides zero cells for clarity.
    - bold title with metrics.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    # 1. Setup the Plot
    fig, ax = plt.subplots(figsize=(16, 16))  # Bigger size

    # 2. Plot with a distinct colormap
    # 'values_format="d"' ensures integers (no scientific notation)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=target_names,
        xticks_rotation=45,  # 45 degree angle is easier to read than 90
        cmap='Blues',
        values_format='d',  # Force integers
        ax=ax,
        colorbar=False
    )

    # 3. VISUAL TWEAK: Remove "0" text to clean up the chart
    # This iterates over the text objects in the matrix and hides zeros
    for text in disp.text_.ravel():
        if text.get_text() == '0':
            text.set_text('')
        else:
            text.set_fontsize(11)  # Make non-zeros bigger
            text.set_weight('bold')

    # 4. Title & Layout
    acc = metrics.get('accuracy', 0.0)
    f1 = metrics.get('macro_f1', 0.0)

    plt.title(
        f"🏆 Best Model Performance\nAccuracy: {acc:.1%} | Macro-F1: {f1:.3f}",
        fontsize=18, fontweight='bold', pad=20
    )
    plt.ylabel("True Label", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')

    # Adjust layout to prevent clipping of long labels
    plt.tight_layout()

    plt.savefig(filepath, dpi=300)  # 300 DPI for high resolution
    plt.close()