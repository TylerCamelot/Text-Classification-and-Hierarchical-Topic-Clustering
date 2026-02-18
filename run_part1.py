import argparse
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data import load_20ng, train_test_split_stratified
from src.part1_classic import get_models, build_vectorizer
from src.utils import save_json
from src.eval import evaluate, top_confusions, confusion, save_confusion_plot
from sklearn.pipeline import Pipeline


def save_results_table(results_dict, filename, title):
    """Saves model metrics to a clean PNG table with robust key checking."""
    data = []
    for key, val in results_dict.items():
        m = val['metrics']

        # Robustly find the F1 keys regardless of exact naming
        acc = m.get('accuracy', 0.0)
        macro = m.get('macro_f1', m.get('macro f1', 0.0))

        data.append([
            val['classifier'],
            val['feature_type'].upper(),
            f"{acc:.4f}",
            f"{macro:.4f}"
        ])

    # Sort by Macro-F1 descending (index 3)
    data.sort(key=lambda x: x[3], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    col_labels = ["Classifier", "Features", "Accuracy", "Macro-F1"]
    table = ax.table(cellText=data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Global seed set to: {seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config()
    set_seeds(cfg.random_state)

    X, y, target_names = load_20ng(cfg)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, cfg)

    feature_types = ["bow", "tfidf"]
    full_results = {}
    best_overall_f1 = -1.0
    best_overall_name = ""
    best_overall_preds = None

    print("\n--- Starting Part 1: Classic Comparison ---")
    for kind in feature_types:
        models = get_models(cfg)
        for name, model in models.items():
            vect = build_vectorizer(cfg, kind=kind)
            pipe = Pipeline([("vect", vect), ("clf", model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            metrics = evaluate(y_test, preds)
            result_key = f"{name}_{kind}"
            full_results[result_key] = {
                "classifier": name, "feature_type": kind, "metrics": metrics
            }

            if metrics['macro_f1'] > best_overall_f1:
                best_overall_f1 = metrics['macro_f1']
                best_overall_name = f"{name} ({kind})"
                best_overall_preds = preds

    # Save outputs
    save_results_table(full_results, os.path.join(args.out_dir, "part1_model_comparison.png"),
                       "Part 1: Classic Model Performance")
    save_confusion_plot(y_test, best_overall_preds, target_names,
                        os.path.join(args.out_dir, "part1_winner_cm.png"), evaluate(y_test, best_overall_preds))
    save_json(full_results, os.path.join(args.out_dir, "part1_results.json"))
    print(f"🏆 Best: {best_overall_name} | Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()