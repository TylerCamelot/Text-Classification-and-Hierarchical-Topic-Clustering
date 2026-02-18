"""Run Part 3 (clustering + hierarchical topic tree) pipeline."""

import argparse
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is on path so "src" resolves correctly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data import load_20ng
from src.llm_labeler import make_labeler
from src.part3_clustering import cluster_and_label, save_tree_plot
from src.utils import save_json

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def plot_elbow(tree_data, filename):
    """Generates a plot of the Elbow Method (Inertia vs Number of Clusters)."""
    ks = tree_data["inertias"]["ks"]
    inertias = tree_data["inertias"]["values"]
    chosen_k = tree_data["k"]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, 'bo-', markerfacecolor='red', markersize=8)
    plt.axvline(x=chosen_k, color='green', linestyle='--', label=f'Chosen K={chosen_k}')
    plt.title("Elbow Method for Optimal K", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia (WCSS)", fontsize=12)
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved Elbow Plot: {filename}")

def save_summary_pdf(tree_data, filename):
    """Saves the cluster summary to PDF with actual Global Row Numbers."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    text_lines = [
        "FINAL TOPIC HIERARCHY & REPRESENTATIVE DOCUMENTS",
        f"Date: 2026-02-17",
        "=" * 55,
        ""
    ]

    sorted_top = sorted(tree_data["top_level"], key=lambda d: -d["size"])
    for node in sorted_top:
        cid = node["cluster_id"]
        text_lines.append(f"CLUSTER [{cid}] - LABEL: {node['label']} (Size: {node['size']})")
        text_lines.append("  Representative Samples:")

        for i in range(min(3, len(node['representatives']))):
            row_idx = node['row_numbers'][i]
            raw_text = node['representatives'][i].replace('\n', ' ').strip()
            # Ensure Cluster 0 shows content by providing a fallback snippet
            snippet = (raw_text[:85] + "..") if len(raw_text) > 5 else "[Non-textual content]"

            text_lines.append(f"    Doc at Row #{row_idx}: {snippet}")

        if cid in tree_data["children"]:
            subs = [s['label'] for s in tree_data["children"][cid]]
            text_lines.append(f"  Sub-topics: {', '.join(subs)}")

        text_lines.append("-" * 65)

    y_pos = 0.95
    for line in text_lines:
        ax.text(0.05, y_pos, line, transform=ax.transAxes,
                fontsize=9, family='monospace', va='top')
        y_pos -= 0.024
        if y_pos < 0.05: break

    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved clean summary PDF with global row mapping: {filename}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config()
    set_seeds(cfg.random_state)

    texts, y, _ = load_20ng(cfg)
    label_fn = make_labeler()

    print("\n--- Running Hierarchical Clustering ---")
    tree = cluster_and_label(texts, cfg, llm_label_fn=label_fn, k=args.k)

    # 1. Save JSON (The int64 fix now allows this to run smoothly)
    save_json(tree, os.path.join(args.out_dir, "topic_tree.json"))

    # 2. Save Visuals
    save_tree_plot(tree, os.path.join(args.out_dir, "topic_tree_viz.png"))
    plot_elbow(tree, os.path.join(args.out_dir, "elbow_plot.png"))
    save_summary_pdf(tree, os.path.join(args.out_dir, "cluster_summary.pdf"))

    print("\n" + "=" * 60)
    print("FINAL TOPIC HIERARCHY & REPRESENTATIVE DOCUMENTS")
    print("=" * 60)
    sorted_top = sorted(tree["top_level"], key=lambda d: -d["size"])
    for node in sorted_top:
        cid = node["cluster_id"]
        print(f"\n📂 CLUSTER [{cid}] - LABEL: {node['label']} (Size: {node['size']})")
        print("   📄 Top Representative Snippets:")
        for i, rep in enumerate(node['representatives'][:3]):
            row_idx = node['row_numbers'][i]
            print(f"      Row #{row_idx}: {rep.replace('\n', ' ').strip()[:110]}...")
        if cid in tree["children"]:
            print("   └── 🌿 Sub-topics:")
            for sub in sorted(tree["children"][cid], key=lambda d: -d["size"]):
                print(f"       ├── {sub['label']} (n={sub['size']})")

if __name__ == "__main__":
    main()