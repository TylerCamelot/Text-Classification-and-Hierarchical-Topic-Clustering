"""LLM-based topic labeling with optional OpenAI or TF-IDF keyword fallback."""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def keyword_fallback_label(representatives: list[str], top_n: int = 3) -> str:
    """Extracts top keywords using TF-IDF when LLM is unavailable or refuses."""
    if not representatives:
        return "Unknown Topic"

    vect = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2),
    )

    try:
        X = vect.fit_transform(representatives)
    except ValueError:
        return "General Topic"

    scores = np.asarray(X.sum(axis=0)).flatten()
    feature_names = vect.get_feature_names_out()
    top_indices = scores.argsort()[::-1][:top_n]

    return " / ".join(feature_names[i] for i in top_indices).title()

def make_labeler():
    """Returns a closure label_fn(reps, level, cluster_id) -> str."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("NOTICE: OPENAI_API_KEY not found. Using keyword extraction fallback.")
        return lambda reps, *args, **kwargs: keyword_fallback_label(reps)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def label_fn(reps: list[str], level: str, cluster_id: any) -> str:
        prompt = (
            "Task: Generate a 2-4 word professional topic label for these snippets.\n"
            "Constraint: Do not explain yourself. Return ONLY the label.\n"
            "Snippets:\n" + "\n".join([f"- {r[:300]}" for r in reps])
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=20
            )
            label = resp.choices[0].message.content.strip().replace('"', '')

            # --- SANITIZER ---
            # Detect if the LLM is talking to the user instead of providing a label
            invalid_phrases = ["please provide", "snippets", "context", "unable to", "i can't", "i do not"]
            is_too_long = len(label.split()) > 6

            if any(phrase in label.lower() for phrase in invalid_phrases) or is_too_long:
                return keyword_fallback_label(reps)

            return label
        except Exception:
            return keyword_fallback_label(reps)

    return label_fn