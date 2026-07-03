# 🍽️ Restaurant Review Sentiment Analysis

End-to-end NLP project that classifies restaurant reviews as **Positive** or
**Negative**, served as a Dockerized [Streamlit](https://streamlit.io/) web app.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![Accuracy](https://img.shields.io/badge/test%20accuracy-82.7%25-success)

## 🚀 demo

<!-- After deploying on Render, paste the URL here -->
![Screenshot 1](screenshots\Screenshot.png)
![Screenshot 2](screenshots\Screenshot2.png)
![Screenshot 3](screenshots\Screenshot3.png)
> Note: the free tier sleeps after inactivity — the first load can take ~30s to wake up.


## 📊 Overview

The model is trained on 1,000 labelled restaurant reviews and predicts customer
sentiment from raw text. The full path is covered: data cleaning → TF-IDF feature
extraction → model selection → serialization → web app → container → cloud deploy.

| Stage | Tooling |
|-------|---------|
| Feature extraction | `TfidfVectorizer` (scikit-learn) |
| Classifier | `SVC` (Support Vector Classifier) |
| Serving | Streamlit |
| Packaging | Docker |
| Hosting | Render (free tier) |

## 🧪 Model selection

Four approaches were compared on a held-out test set (22% split). TF-IDF features
outperformed raw counts, and the **TF-IDF + SVC pipeline** won:

| Model | Vectorizer | Test accuracy |
|-------|------------|---------------|
| **SVC (pipeline)** ✅ | **TF-IDF** | **82.7%** |
| MultinomialNB (pipeline) | TF-IDF | 81.4% |
| MultinomialNB (pipeline) | CountVectorizer | 80.5% |
| SVC (pipeline) | CountVectorizer | 80.0% |

Full classification report (test set):

```
              precision    recall  f1-score   support
    Negative       0.81      0.84      0.82       106
    Positive       0.85      0.82      0.83       114
    accuracy                           0.83       220
```

## 🗂️ Project structure

```
sentiment-analysis/
├── train.py                 # trains TF-IDF + SVC, saves model/sentiment_pipeline.joblib
├── app.py                   # Streamlit web app
├── requirements.txt         # pinned dependencies
├── Dockerfile               # builds the container (trains model at build time)
├── .dockerignore
├── Restaurant_Reviews.tsv   # dataset (1,000 reviews)
└── NLP_PROJECT.ipynb        # original exploratory notebook
```

## 🏃 Run locally (without Docker)

```bash
git clone https://github.com/dscoder001/SENTIMENT-ANALYSIS.git
cd SENTIMENT-ANALYSIS

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py            # trains the model -> model/sentiment_pipeline.joblib
streamlit run app.py       # opens http://localhost:8501
```

## 🐳 Run with Docker

The image trains the model during the build, so it ships ready to serve.

```bash
# build the image
docker build -t sentiment-app .

# run it, mapping host port 8501 -> container port 8501
docker run -p 8501:8501 sentiment-app
```

Then open **http://localhost:8501** in your browser.

> `--server.address=0.0.0.0` in the Dockerfile is what makes the app reachable
> from your host browser — without it, Streamlit only listens inside the container.

## ☁️ Deploy on Render

Render reads the `Dockerfile` directly — no extra config needed.

1. Push this repo to GitHub.
2. On [render.com](https://render.com): **New + → Web Service**, select this repo.
3. Render auto-detects the Dockerfile. Instance type: **Free**.
4. **Create Web Service** — first build takes 3–5 minutes.
5. Copy the public URL into the **Live demo** section above.

## 🛠️ Tech stack

`Python` · `pandas` · `scikit-learn` · `Streamlit` · `Docker` · `Render`

## 👤 Author

**Dhiman Saha** — [GitHub](https://github.com/dscoder001)

Dataset: [Restaurant Reviews (Kaggle)](https://www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews)
