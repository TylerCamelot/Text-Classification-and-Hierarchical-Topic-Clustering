"""Part 2: Embedding-based text classification with Sentence-Transformers."""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from .config import Config
from .eval import confusion, evaluate, top_confusions

# Dimension for fallback embeddings (match MiniLM-L6-v2 output size).
_FALLBACK_DIM = 384


def _embed_fallback(texts, cfg: Config, fit_tuple=None):
    """
    TF-IDF + TruncatedSVD (LSA) fallback when Sentence-Transformers is missing.
    """
    # Convert generator to list if needed
    texts = list(texts)

    # Training Mode: Fit vectorizer and SVD
    if fit_tuple is None:
        vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
        X = vect.fit_transform(texts)

        # Reduce dims to simulate embeddings
        svd = TruncatedSVD(
            n_components=min(_FALLBACK_DIM, X.shape[1] - 1, X.shape[0] - 1),
            random_state=cfg.random_state
        )
        emb = svd.fit_transform(X)
        return np.asarray(emb, dtype=np.float64), (vect, svd)

    # Inference Mode: Use existing vectorizer and SVD
    vect, svd = fit_tuple
    X = vect.transform(texts)
    emb = svd.transform(X)
    return np.asarray(emb, dtype=np.float64), fit_tuple


def embed_train_test(X_train, X_test, cfg: Config):
    """
    Embeds train and test sets. Tries SentenceTransformer first, falls back to SVD.
    """
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading SentenceTransformer: {cfg.st_model_name}...")
        model = SentenceTransformer(cfg.st_model_name)

        # Encode both sets (no fitting required for pre-trained models)
        Xtr = model.encode(list(X_train), batch_size=cfg.batch_size, show_progress_bar=True)
        Xte = model.encode(list(X_test), batch_size=cfg.batch_size, show_progress_bar=True)
        return np.array(Xtr), np.array(Xte)

    except ImportError:
        print("SentenceTransformers not found/failed. Using TF-IDF+SVD fallback.")
        # Fit fallback on Train ONLY, then apply to Test
        Xtr, fit_tuple = _embed_fallback(X_train, cfg, None)
        Xte, _ = _embed_fallback(X_test, cfg, fit_tuple)
        return Xtr, Xte


def get_models(cfg: Config):
    return {
        "MNB": MultinomialNB(),

        # REMOVED n_jobs=-1 here to silence the warning
        "LogReg": LogisticRegression(max_iter=1000),

        # FIX: Increase max_iter to 10000 to solve ConvergenceWarning
        "LinearSVM": LinearSVC(dual="auto", max_iter=10000),

        # KEEP n_jobs=-1 here
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=cfg.random_state,
            n_jobs=-1,
        ),
    }


def run_part2(X_train_text, X_test_text, y_train, y_test, cfg: Config, target_names=None):
    """
    Main execution for Part 2.
    """
    # 1. Generate Embeddings
    print("Generating embeddings for Part 2...")
    X_train_emb, X_test_emb = embed_train_test(X_train_text, X_test_text, cfg)

    # 2. Prepare Scaled Version for MNB (Fixes Leakage)
    # MinMaxScaler fits on TRAIN, transforms TEST. Result is [0, 1].
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_emb)
    X_test_scaled = scaler.transform(X_test_emb)

    results = {}
    for name, model in get_models(cfg).items():
        print(f"Training {name} on embeddings...")

        # Select correct data version
        if name == "MNB":
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train_emb, X_test_emb

        model.fit(X_tr, y_train)
        preds = model.predict(X_te)

        metrics = evaluate(y_test, preds)
        cm = confusion(y_test, preds)
        conf_pairs = top_confusions(cm, target_names=target_names, top_n=10)

        results[name] = {
            "embedding_model": cfg.st_model_name,
            "metrics": metrics,
            "top_confusions": conf_pairs,
        }

    return results