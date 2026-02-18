"""Configuration and environment for nlp-topic-tree.

Central dataclass holding all pipeline hyperparameters and flags.
Frozen so config cannot be mutated after creation.
"""

from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    """Immutable pipeline configuration.

    Covers data loading, Part 1 (classic), Part 2 (embeddings),
    and Part 3 (clustering) settings.
    """

    # --- Reproducibility & Data ---
    # Seed for train/test split and any stochastic steps (e.g., KMeans, RandomForest).
    random_state: int = 42
    # Fraction of data reserved for evaluation (e.g. 20%).
    test_size: float = 0.2
    # Cap on number of rows to load (Requirement: 10k rows).
    max_rows: int = 10000
    # Whether to strip boilerplate (headers, footers, block quotes) from documents.
    # CRITICAL: 20Newsgroups contains metadata that allows models to cheat.
    remove_headers_footers_quotes: bool = True

    # --- Part 1: Classic Vectorization (TF-IDF / BOW) ---
    # Max vocabulary size for TF-IDF/BoW vectorizers.
    max_features: int = 50000
    # (min_n, max_n) for character/word n-grams (e.g. (1, 2) = unigrams + bigrams).
    ngram_range: tuple[int, int] = (1, 2)
    # Ignore terms that appear in fewer than this many documents.
    min_df: int = 2

    # --- Part 2: Embeddings ---
    # Sentence-Transformers model name for encoding documents.
    # 'all-MiniLM-L6-v2' is fast and effective for clustering.
    st_model_name: str = "all-MiniLM-L6-v2"
    # Batch size when computing embeddings.
    batch_size: int = 64
    # MultinomialNB cannot handle negative values (which embeddings have).
    # If True, apply MinMaxScaler (0-1) to embeddings before MNB training.
    scale_embeddings: bool = True

    # --- Part 3: Clustering & Topic Tree ---
    # --- Part 3: Clustering & Topic Tree ---
    # Minimum number of clusters to try (e.g. for elbow/silhouette search).
    k_min: int = 2

    # Maximum number of clusters; must be < 10 per requirement.
    k_max: int = 9

    # Requirement: The 2 largest clusters must be re-clustered into exactly 3 sub-clusters.
    n_sub_clusters: int = 3

    # Number of representative documents (closest to centroid) to send to LLM.
    top_k_docs_for_context: int = 5

    # LLM Model Name (e.g., 'gpt-3.5-turbo', 'gpt-4o-mini').
    llm_model_name: str = "gpt-3.5-turbo"