import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

os.makedirs("models", exist_ok=True)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(max_iter=1000, C=4)),
])

print("Training...")
start = time.time()
pipe.fit(train["text"], train["label"])
print(f"Training time: {time.time() - start:.1f}s")

preds = pipe.predict(test["text"])
print(f"\nFull test set accuracy (25,000 reviews): {accuracy_score(test['label'], preds):.4f}")
print(classification_report(test["label"], preds, target_names=["negative", "positive"]))

joblib.dump(pipe, "models/tfidf_lr.joblib")
print("Saved model to models/tfidf_lr.joblib")