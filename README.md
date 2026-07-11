# Cyber-Bullying Tweet Detector

A Streamlit web app for detecting cyberbullying in tweets using a trained XGBoost classifier.

The project preprocesses tweet text, transforms it with a TF-IDF vectorizer, and predicts one of the following categories:

- `gender` (gender-based bullying)
- `religion` (religion-based bullying)
- `age` (age-based bullying)
- `ethnicity` (ethnicity-based bullying)
- `other_cyberbullying` (other bullying content)
- `not_cyberbullying` (not cyberbullying)

## Features

- Clean and normalize tweet text
- Remove URLs, mentions, hashtags, punctuation, numbers, and stop words
- Lemmatize tokens using NLTK
- Predict cyberbullying category with model confidence visualization
- Display model metadata and metrics

## Model Information

- **Model:** XGBoost Classifier
- **Accuracy:** 83.52%
- **F1 Macro:** 0.8316

These values are loaded from `model_metadata.json`.

## Live Demo

[![Live Deployment](https://img.shields.io/badge/Status-Live_on_Render-success)](https://cyber-bullying-tweet-detector.onrender.com)


## Getting Started

### Prerequisites

- Python 3.10+ (recommended)
- `pip` package manager

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Repository Files

- `app.py` - Streamlit application entry point
- `requirements.txt` - Python dependencies
- `best_model.pkl` - serialized trained model
- `tfidf_vectorizer.pkl` - serialized TF-IDF vectorizer
- `label_encoder.pkl` - serialized label encoder
- `model_metadata.json` - metadata for the trained model

## Notes

- The app downloads NLTK resources for stop words and lemmatization at startup.
- If the NLTK data is not already installed, the app will fetch it automatically.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
