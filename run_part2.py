import argparse
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data import load_20ng, train_test_split_stratified
from src.part2_embeddings import embed_train_test, get_models
from src.utils import save_json
from src.eval import evaluate, confusion, save_confusion_plot
from sklearn.preprocessing import MinMaxScaler


def save_embedding_results_table(results_dict, filename, model_name):
    """Saves embedding classifier metrics to a PNG table with robust key handling."""
    data = []
    for name, val in results_dict.items():
        m = val['metrics']

        # Robustly pull metrics using .get() to avoid KeyErrors
        acc = m.get('accuracy', 0.0)
        macro = m.get('macro_f1', m.get('macro f1', 0.0))

        data.append([
            name,
            f"{acc:.4f}",
            f"{macro:.4f}"
        ])

    # Sort by Macro-F1 descending (index 2)
    data.sort(key=lambda x: x[2], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    col_labels = ["Classifier", "Accuracy", "Macro-F1"]
    table = ax.table(cellText=data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title(f"Part 2: Embedding Classifiers ({model_name})",
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved embedding results table to: {filename}")

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
    X_train_text, X_test_text, y_train, y_test = train_test_split_stratified(X, y, cfg)

    print(f"\n--- Generating Embeddings ({cfg.st_model_name}) ---")
    X_train_emb, X_test_emb = embed_train_test(X_train_text, X_test_text, cfg)

    models = get_models(cfg)
    full_results = {}
    best_f1, best_name, best_preds = -1.0, "", None

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_emb)
    X_test_scaled = scaler.transform(X_test_emb)

    for name, model in models.items():
        X_tr, X_te = (X_train_scaled, X_test_scaled) if name == "MNB" else (X_train_emb, X_test_emb)
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        metrics = evaluate(y_test, preds)

        full_results[name] = {"metrics": metrics}
        if metrics['macro_f1'] > best_f1:
            best_f1, best_name, best_preds = metrics['macro_f1'], name, preds

    # Save outputs
    save_embedding_results_table(full_results, os.path.join(args.out_dir, "part2_model_comparison.png"),
                                 cfg.st_model_name)
    save_confusion_plot(y_test, best_preds, target_names,
                        os.path.join(args.out_dir, "part2_winner_cm.png"), evaluate(y_test, best_preds))
    save_json(full_results, os.path.join(args.out_dir, "part2_results.json"))
    print(f"🏆 Best: {best_name} | Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()