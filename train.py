"""
Train the restaurant-review sentiment classifier.

Pipeline: TF-IDF  ->  Support Vector Classifier (SVC)
This is the best model from the original analysis (~82.7% test accuracy).

Running this script produces a single serialized artifact at
`model/sentiment_pipeline.joblib`, which the Streamlit app loads at runtime.
Training inside the Docker build keeps the model and the runtime library
versions perfectly in sync (no joblib cross-version warnings).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

DATA_PATH = Path("Restaurant_Reviews.tsv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "sentiment_pipeline.joblib"

# Reproduces the original notebook split exactly.
TEST_SIZE = 0.22
RANDOM_STATE = 11


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Drop exact duplicate reviews, keeping the last occurrence.
    df = df.drop_duplicates(keep="last")
    return df


def main() -> None:
    df = load_data(DATA_PATH)
    X = df["Review"].values
    y = df["Liked"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # probability=True enables predict_proba so the app can show a
    # confidence score; it does not change the decision boundary.
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SVC(probability=True, random_state=RANDOM_STATE),
    )

    print("Training TF-IDF + SVC pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
