"""Part 3: Clustering and hierarchical topic structure."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from .config import Config
from .utils import shorten

def get_embeddings_part3(texts, cfg: Config):
    """Generates normalized embeddings for clustering."""
    print("Part 3: Encoding documents for clustering...")
    model = SentenceTransformer(cfg.st_model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.array(embeddings)

def elbow_inertia(X, cfg: Config):
    """Calculates inertia for the Elbow Method."""
    inertias = []
    max_k = min(cfg.k_max, X.shape[0] - 1)
    ks = list(range(cfg.k_min, max_k + 1))

    print(f"Running Elbow Method for k={ks[0]}..{ks[-1]}...")
    for k in ks:
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init="auto")
        km.fit(X)
        inertias.append(float(km.inertia_))
    return ks, inertias

def choose_k_by_elbow(ks, inertias):
    """Heuristic to find the 'bend' in the inertia curve."""
    if len(inertias) < 2:
        return ks[0]
    drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    rel_drops = [drops[i] / inertias[i] for i in range(len(drops))]

    if not rel_drops:
        return ks[-1]
    best_i = int(np.argmax(rel_drops))
    return ks[best_i + 1]

def closest_docs_to_centroid(X, labels, centroids, texts, cluster_id, top_n=8):
    """Identifies representative documents and their ORIGINAL row indices."""
    idx = np.where(labels == cluster_id)[0]
    Xc = X[idx]
    if len(Xc) == 0: return [], []

    centroid = centroids[cluster_id].reshape(1, -1)
    dists = cosine_distances(Xc, centroid).reshape(-1)
    local_best_indices = np.argsort(dists)[:top_n]

    # FIX: Explicitly cast to Python int to avoid JSON int64 serializability error
    global_indices = [int(idx[i]) for i in local_best_indices]

    reps = [texts[i] for i in global_indices]
    return reps, global_indices

def cluster_and_label(texts, cfg: Config, llm_label_fn, k=None):
    """Executes hierarchical clustering and labels results."""
    X = get_embeddings_part3(texts, cfg)
    ks, inertias = elbow_inertia(X, cfg)
    if k is None:
        k = choose_k_by_elbow(ks, inertias)
    k = min(k, cfg.k_max)

    print(f"Clustering top-level with k={k}...")
    km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init="auto")
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    top_level_clusters = []
    counts = [(int((labels == i).sum()), i) for i in range(k)]
    counts.sort(reverse=True)

    for size, cid in counts:
        print(f"  Labeling Cluster {cid} (size: {size})...")
        reps, indices = closest_docs_to_centroid(X, labels, centroids, texts, cid, cfg.top_k_docs_for_context)
        label = llm_label_fn(reps, level="top", cluster_id=cid)
        top_level_clusters.append({
            "cluster_id": int(cid),
            "size": int(size),
            "label": label,
            "representatives": reps,
            "row_numbers": indices
        })

    biggest_two_ids = [cid for _, cid in counts[:2]]
    children = {}

    print("Sub-clustering the 2 largest groups...")
    for cid in biggest_two_ids:
        member_indices = np.where(labels == cid)[0]
        X_sub = X[member_indices]
        texts_sub = [texts[i] for i in member_indices]

        sub_k = cfg.n_sub_clusters
        km_sub = KMeans(n_clusters=sub_k, random_state=cfg.random_state, n_init="auto")
        sub_labels = km_sub.fit_predict(X_sub)
        sub_centroids = km_sub.cluster_centers_

        subs = []
        for sub_id in range(sub_k):
            reps, sub_indices_local = closest_docs_to_centroid(X_sub, sub_labels, sub_centroids, texts_sub, sub_id,
                                               top_n=cfg.top_k_docs_for_context)

            # FIX: Explicitly cast sub-cluster row numbers to Python int
            sub_row_numbers = [int(member_indices[i]) for i in sub_indices_local]

            sub_label = llm_label_fn(reps, level="sub", cluster_id=f"{cid}.{sub_id}")
            subs.append({
                "subcluster_id": int(sub_id),
                "size": int((sub_labels == sub_id).sum()),
                "label": sub_label,
                "representatives": reps,
                "row_numbers": sub_row_numbers
            })
        children[int(cid)] = subs

    return {
        "k": int(k),
        "inertias": {"ks": [int(k_val) for k_val in ks], "values": inertias},
        "top_level": top_level_clusters,
        "children": children,
    }

def save_tree_plot(tree_data, filename):
    """Visualizes the tree structure and exports to PNG with anti-overlap spacing."""
    top_nodes = sorted(tree_data['top_level'], key=lambda x: x['size'], reverse=True)
    n_top = len(top_nodes)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    ax.set_ylim(-0.1, 1.1)

    root_pos = (0.5, 0.95)
    layer1_y = 0.65
    layer2_y = 0.20

    ax.text(root_pos[0], root_pos[1], "Full Dataset\n(10k docs)",
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round", fc="#dddddd", pad=0.8))

    x_steps = np.linspace(0.1, 0.9, n_top)

    for i, node in enumerate(top_nodes):
        x = x_steps[i]
        cid, lbl, size = node['cluster_id'], node['label'], node['size']
        has_children = cid in tree_data['children']

        ax.plot([root_pos[0], x], [root_pos[1] - 0.05, layer1_y + 0.05], 'k-', lw=1, alpha=0.3)

        box_color = "skyblue" if has_children else "white"
        display_lbl = (lbl[:22] + '..') if len(lbl) > 22 else lbl
        ax.text(x, layer1_y, f"C{cid}\n{display_lbl}\n(n={size})",
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round", fc=box_color, ec="blue" if has_children else "gray", alpha=0.9))

        if has_children:
            subs = tree_data['children'][cid]
            sub_xs = np.linspace(x - 0.18, x + 0.18, len(subs))
            for j, sub in enumerate(subs):
                sx = sub_xs[j]
                ax.plot([x, sx], [layer1_y - 0.05, layer2_y + 0.05], 'k--', lw=1, alpha=0.4)
                s_disp = (sub['label'][:18] + '..') if len(sub['label']) > 18 else sub['label']
                ax.text(sx, layer2_y, f"{s_disp}\n(n={sub['size']})",
                        ha='right', va='top', fontsize=8, rotation=35,
                        bbox=dict(boxstyle="round", fc="#fffacd", ec="orange", alpha=0.9))

    plt.title(f"Topic Clustering Tree (k={tree_data['k']})", fontsize=18, fontweight='bold', pad=60)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()