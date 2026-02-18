import pandas as pd
from src.config import Config
from src.data import load_20ng


def main():
    cfg = Config()
    # Load the actual data being used in your project
    texts, labels, target_names = load_20ng(cfg)

    # Create a DataFrame for a clean UI view in PyCharm
    df = pd.DataFrame({
        "Category": [target_names[i] for i in labels],
        "Snippet": [t[:200] + "..." for t in texts],
        "Full_Text": texts
    })

    # This line is where you pause in your video to show the data
    print(df.head(20))


if __name__ == "__main__":
    main()