# Sentiment Analysis Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dscoder001/SENTIMENT-ANALYSIS/blob/main/NLP_PROJECT.ipynb)

## Overview

This project focuses on sentiment analysis using Natural Language Processing (NLP) techniques. It involves analyzing and classifying the sentiment of restaurant reviews.

## Author

**Dhiman Saha**

## Environment

The project is developed and tested in Google Colab.

## Libraries Used

- **NumPy** (`np`): For numerical operations.
- **Pandas** (`pd`): For handling datasets.
- **Scikit-learn**:
  - `train_test_split`: To train and split the data.
  - `TfidfVectorizer`: To create the TF-IDF vectors.
  - `Support Vector Classifier (SVC)`: For classification.
  - `make_pipeline`: For creating pipelines.
  - `MultinomialNB`: For Naive Bayes classification.

## Project Structure

1. **Importing Libraries**: Initial setup and importing necessary libraries.
2. **Reading Data**: Loading and preparing the dataset (`job_postings_raw.dat`).
3. **Data Analysis**: Identifying patterns and insights in the data.
4. **Feature Engineering**: Creating TF-IDF vectors.
5. **Model Training**: Training models using SVC and Naive Bayes classifiers.
6. **Evaluation**: Evaluating the performance of the models.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/dscoder001/SENTIMENT-ANALYSIS.git
2. Open the project in Google Colab by clicking the "Open In Colab" badge above.
3. Run the cells in the notebook sequentially to reproduce the results.
   
## Results
Description of results (e.g., accuracy, precision, recall) obtained from the models.
Visualizations and insights derived from the data analysis.
## Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
