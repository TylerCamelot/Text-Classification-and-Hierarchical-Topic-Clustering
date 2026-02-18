import json
from pathlib import Path


def save_json(obj, path: str):
    # Create parent directories if needed, then write obj as pretty-printed JSON.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def shorten(text: str, n=280):
    # Collapse whitespace and truncate to n chars, appending "..." if truncated.
    # Handle None and collapse internal whitespace to a single space.
    text = " ".join((text or "").split())
    return text[:n] + ("..." if len(text) > n else "")
