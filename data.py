"""Data loading and preprocessing for 20 Newsgroups."""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# Assumes config.py is in the same directory
from .config import Config


def load_20ng(cfg: Config):
    """
    Loads 20 Newsgroups, optionally strips metadata, and downsamples
    to a specific size while maintaining class balance.
    """
    # 1. Setup removal tuple based on config
    remove = ()
    if cfg.remove_headers_footers_quotes:
        remove = ("headers", "footers", "quotes")

    # 2. Fetch full dataset (train + test combined)
    print("Fetching dataset...")
    dataset = fetch_20newsgroups(subset="all", remove=remove)

    X = np.array(dataset.data, dtype=object)
    y = np.array(dataset.target)
    target_names = dataset.target_names

    # 3. Stratified Subsampling (The Fix)
    # If the dataset is larger than max_rows, downsample it while keeping class ratios.
    if len(X) > cfg.max_rows:
        print(f"Downsampling from {len(X)} to {cfg.max_rows} rows (stratified)...")
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=cfg.max_rows,
            stratify=y,
            random_state=cfg.random_state
        )

    return X, y, target_names


def train_test_split_stratified(X, y, cfg: Config):
    """
    Split X, y into train/test with stratification using config settings.
    """
    return train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )