import sys
import re
import pickle


MODEL_PATH = "models/nb_pipeline.pkl"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<:[a-zA-Z0-9_]+:\d+>", "", text)
    text = re.sub(r"<[@#][!&]?\d+>", "", text)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def predict(message: str) -> tuple[str, float]:
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    vectorizer = pipeline["vectorizer"]
    model = pipeline["model"]

    cleaned = clean_text(message)
    if not cleaned:
        raise ValueError("Message is empty after cleaning.")

    X = vectorizer.transform([cleaned])
    author = model.predict(X)[0]
    confidence = model.predict_proba(X).max()
    return author, float(confidence)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "your message here"')
        sys.exit(1)

    message = " ".join(sys.argv[1:])
    try:
        author, confidence = predict(message)
        print(f"Predicted: {author} (confidence: {confidence:.2f})")
    except FileNotFoundError:
        print(f"Error: model not found at {MODEL_PATH}. Run the notebook first.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
