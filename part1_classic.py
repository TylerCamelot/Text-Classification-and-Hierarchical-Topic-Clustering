"""Part 1: Classic text classification with BOW/TF-IDF and multiple classifiers."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import Config
from .eval import confusion, evaluate, top_confusions


def build_vectorizer(cfg: Config, kind: str):
    """
    Returns a Vectorizer based on the config settings.
    Unified max_features handles both BOW and TF-IDF.
    """
    if kind == "bow":
        return CountVectorizer(
            max_features=cfg.max_features,  # Updated to match Config
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
            stop_words="english",
        )
    if kind == "tfidf":
        return TfidfVectorizer(
            max_features=cfg.max_features,  # Updated to match Config
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
            stop_words="english",
        )
    raise ValueError(f"Vectorizer kind '{kind}' not supported. Use 'bow' or 'tfidf'.")


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


def run_part1(X_train, X_test, y_train, y_test, cfg: Config, target_names=None, kind="tfidf"):
    """
    Train each classifier in a pipeline with the chosen vectorizer.
    Returns dictionary of metrics and confusion data.
    """
    results = {}

    # Note: We rebuild the vectorizer for every model. While slightly inefficient
    # (vectorizing 4 times), it strictly adheres to the "Pipeline(Vector -> Model)"
    # requirement, ensuring zero leakage and easy extensibility.

    for name, model in get_models(cfg).items():
        print(f"Training {name} with {kind}...")

        # Create fresh vectorizer instance for each pipeline
        vect = build_vectorizer(cfg, kind=kind)

        pipe = Pipeline([
            ("vect", vect),
            ("clf", model),
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        metrics = evaluate(y_test, preds)
        cm = confusion(y_test, preds)
        conf_pairs = top_confusions(cm, target_names=target_names, top_n=10)

        results[name] = {
            "vectorizer": kind,
            "metrics": metrics,
            "top_confusions": conf_pairs,
        }

    return results