"""Mini AI project: Sentiment Analyzer using scikit-learn (TF-IDF + Logistic Regression)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

DATA = [
    ("I love this product, it works great", 1),
    ("Absolutely fantastic experience", 1),
    ("The service was excellent and fast", 1),
    ("Really happy with my purchase", 1),
    ("This is amazing, highly recommend", 1),
    ("Great quality and good price", 1),
    ("What a wonderful day", 1),
    ("I am very satisfied with the results", 1),
    ("Best decision I have made", 1),
    ("Super helpful and friendly staff", 1),
    ("I hate this, it is terrible", 0),
    ("Worst experience ever", 0),
    ("The product broke after one day", 0),
    ("Very disappointed and angry", 0),
    ("Awful service, never coming back", 0),
    ("Poor quality and waste of money", 0),
    ("This is the most boring movie", 0),
    ("I am unhappy with the delay", 0),
    ("Terrible support, no one helped", 0),
    ("Bad, slow and useless", 0),
]


def train():
    texts, labels = zip(*DATA)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(C=10))
    model.fit(texts, labels)
    print("Model trained on", len(texts), "sentences.")
    return model


def main():
    model = train()
    print("Type a sentence to analyze its sentiment (or 'quit').")
    while True:
        text = input("> ").strip()
        if text.lower() in {"quit", "exit", ""}:
            break
        prob = model.predict_proba([text])[0][1]
        label = "Positive 😊" if prob >= 0.5 else "Negative 😞"
        print(f"{label} (confidence: {max(prob, 1 - prob):.0%})")


if __name__ == "__main__":
    main()
