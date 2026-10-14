# Sentiment Analyzer (Python + AI)

A mini machine-learning project that classifies text as positive or negative
using TF-IDF features and Logistic Regression (scikit-learn).

## Run
```bash
pip install -r requirements.txt
python main.py
```

## How it works
1. Small labeled dataset of positive/negative sentences
2. TF-IDF converts text to numeric vectors
3. Logistic Regression learns to predict sentiment
4. Interactive CLI to test your own sentences

## Ideas to extend
- Load a larger dataset (e.g. IMDB reviews)
- Save the model with joblib
- Add a Streamlit web UI
