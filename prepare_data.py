import os
import pandas as pd
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

print("Downloading IMDB dataset...")
ds = load_dataset("stanfordnlp/imdb")

train = pd.DataFrame(ds["train"])
test = pd.DataFrame(ds["test"])

for df in (train, test):
    df["text"] = df["text"].str.replace("<br />", " ", regex=False)

print("Train size:", len(train), "| Test size:", len(test))
print(train["label"].value_counts())

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

neg = test[test["label"] == 0].sample(100, random_state=42)
pos = test[test["label"] == 1].sample(100, random_state=42)
sample = pd.concat([neg, pos]).sample(frac=1, random_state=42).reset_index(drop=True)
sample.to_csv("data/eval_sample.csv", index=False)

print("Saved train.csv, test.csv and eval_sample.csv (200 reviews)")