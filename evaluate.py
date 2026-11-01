import argparse
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from llm_utils import STRATEGIES, classify, get_client, get_model

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200, help="number of reviews to evaluate")
args = parser.parse_args()

os.makedirs("results", exist_ok=True)
df = pd.read_csv("data/eval_sample.csv").head(args.n)
y_true = df["label"].values
rows = []

# ---------- Classic model ----------
model = joblib.load("models/tfidf_lr.joblib")
preds, lats = [], []
for text in df["text"]:
    t = time.perf_counter()
    p = int(model.predict([text])[0])
    lats.append(time.perf_counter() - t)
    preds.append(p)

rows.append({
    "method": "TF-IDF + LogReg",
    "accuracy": accuracy_score(y_true, preds),
    "macro_f1": f1_score(y_true, preds, average="macro", labels=[0, 1]),
    "avg_latency_s": float(np.mean(lats)),
    "avg_tokens": 0,
    "cost_per_1000_usd": 0.0,
    "unparsed": 0,
})

# ---------- LLM strategies ----------
client = get_client()
print("Using LLM model:", get_model(client))

for strat in STRATEGIES:
    path = f"results/{strat}_{args.n}.csv"
    if os.path.exists(path):
        print(f"Loading cached results for {strat}")
        res = pd.read_csv(path)
    else:
        out = []
        for text in tqdm(df["text"], desc=strat):
            out.append(classify(text, strat, client=client))
            # save progress so a crash doesn't lose everything
            pd.DataFrame(out).to_csv(f"results/{strat}_{args.n}_partial.csv", index=False)
        res = pd.DataFrame(out)
        res.to_csv(path, index=False)

    p = res["pred"].values
    res["raw"] = res["raw"].fillna("")
    wrong = df.assign(pred=p, raw=res["raw"].values)[p != y_true]
    wrong.to_csv(f"results/errors_{strat}_{args.n}.csv", index=False)

    rows.append({
        "method": f"LLM {strat}",
        "accuracy": accuracy_score(y_true, p),
        "macro_f1": f1_score(y_true, p, average="macro", labels=[0, 1]),
        "avg_latency_s": res["latency_s"].mean(),
        "avg_tokens": (res["in_tokens"] + res["out_tokens"]).mean(),
        "cost_per_1000_usd": res["cost_usd"].mean() * 1000,
        "unparsed": int((p == -1).sum()),
    })

summary = pd.DataFrame(rows)
summary.to_csv("results/summary.csv", index=False)
print("\n", summary.round(4).to_string(index=False))

# ---------- Charts ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
names = summary["method"]

axes[0].bar(names, summary["accuracy"] * 100)
axes[0].set_title("Accuracy (%)")
axes[0].set_ylim(50, 100)

axes[1].bar(names, summary["avg_latency_s"])
axes[1].set_title("Avg latency per review (s, log scale)")
axes[1].set_yscale("log")

axes[2].bar(names, summary["avg_tokens"])
axes[2].set_title("Avg tokens per review")

for ax in axes:
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("results/comparison.png", dpi=150)
print("Saved results/summary.csv and results/comparison.png")