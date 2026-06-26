from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".." / ".." / "dataset" / "training_data.csv"
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"


def train_model(data_path: Path = DATA_PATH) -> Pipeline:
    import csv

    texts, labels = [], []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["sentiment"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(C=1.0)),
    ])
    model.fit(texts, labels)
    return model


def save_model(model_path: Path = MODEL_PATH, data_path: Path = DATA_PATH) -> Path:
    model = train_model(data_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


if __name__ == "__main__":
    path = save_model()
    print(f"Saved model to {path}")
