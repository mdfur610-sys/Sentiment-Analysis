import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://api.groq.com/openai/v1"

# Models are tried in this order; the first one your key can access is used.
# You can force one by adding GROQ_MODEL=model-name to your .env file.
PREFERRED_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
NOT_CHAT = ("whisper", "tts", "guard", "playai", "embed", "orpheus",
            "safeguard", "allam")

# Free tier = $0. Set real prices here if you move to a paid plan.
PRICE_INPUT_PER_M = 0.0
PRICE_OUTPUT_PER_M = 0.0

MAX_CHARS = 1500            # shorter reviews = fewer tokens = fewer rate-limit problems
STRATEGIES = ["zero_shot", "few_shot", "cot"]
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "5"))

SYSTEM = "You are a precise sentiment classifier for movie reviews."

ZERO_SHOT = """Classify the sentiment of the movie review below as positive or negative.
Answer with exactly one word: positive or negative.

Review:
{review}"""

FEW_SHOT = """Classify the sentiment of each movie review as positive or negative.
Answer with exactly one word.

Review: An absolute masterpiece. The acting was superb and I was moved to tears.
Sentiment: positive

Review: Two hours of my life I will never get back. Flat characters and a ridiculous plot.
Sentiment: negative

Review: Not perfect, but the soundtrack and the lead performance made it worth watching.
Sentiment: positive

Review: I expected a thriller, but it was slow, predictable and badly edited.
Sentiment: negative

Review: {review}
Sentiment:"""

COT = """Classify the sentiment of the movie review below as positive or negative.
First, think step by step in 2-4 short sentences: note the key opinions and watch for sarcasm, mixed feelings, or praise and criticism of different aspects.
Then, on the last line, write exactly one of:
Final answer: positive
Final answer: negative

Review:
{review}"""

TEMPLATES = {"zero_shot": ZERO_SHOT, "few_shot": FEW_SHOT, "cot": COT}

_model_cache = {}


def get_client(api_key=None):
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("No API key found. Set GROQ_API_KEY in .env or pass one in.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def is_reasoning(model):
    m = model.lower()
    return "gpt-oss" in m or "qwen3" in m or "deepseek-r1" in m


def get_model(client):
    """Pick a working chat model for this key (cached)."""
    forced = os.getenv("GROQ_MODEL")
    if forced:
        return forced
    cache_key = id(client)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    available = [m.id for m in client.models.list().data]
    chat = [m for m in available if not any(x in m.lower() for x in NOT_CHAT)]
    chosen = next((m for m in PREFERRED_MODELS if m in chat), None)
    if chosen is None:
        if not chat:
            raise RuntimeError("No chat models available for this key. Run list_models.py.")
        chosen = chat[0]
    _model_cache[cache_key] = chosen
    return chosen


def max_tokens_for(strategy, model):
    if is_reasoning(model):
        # reasoning models spend tokens thinking before they answer
        return {"zero_shot": 500, "few_shot": 500, "cot": 1000}[strategy]
    return {"zero_shot": 10, "few_shot": 10, "cot": 300}[strategy]


def parse_label(text, strategy):
    """Return 1 (positive), 0 (negative) or -1 (could not parse)."""
    t = (text or "").strip().lower()
    if strategy == "cot":
        found = re.findall(r"final answer:\s*\**\s*(positive|negative)", t)
        if found:
            return 1 if found[-1] == "positive" else 0
        return -1
    if t.startswith("positive"):
        return 1
    if t.startswith("negative"):
        return 0
    found = re.findall(r"\b(positive|negative)\b", t)
    if found:
        return 1 if found[-1] == "positive" else 0
    return -1


def cost_usd(in_tokens, out_tokens):
    return (in_tokens / 1e6) * PRICE_INPUT_PER_M + (out_tokens / 1e6) * PRICE_OUTPUT_PER_M


def classify(review, strategy, client=None, retries=6):
    client = client or get_client()
    model = get_model(client)
    prompt = TEMPLATES[strategy].format(review=review[:MAX_CHARS])

    kwargs = dict(
        model=model,
        max_tokens=max_tokens_for(strategy, model),
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    if "gpt-oss" in model.lower():
        kwargs["extra_body"] = {"reasoning_effort": "low"}

    resp = None
    latency = 0.0
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            resp = client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - start
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))  # back off if rate-limited

    raw = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    time.sleep(SLEEP_BETWEEN_CALLS)  # stay under the free rate limits
    return {
        "pred": parse_label(raw, strategy),
        "raw": raw,
        "latency_s": latency,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "cost_usd": cost_usd(in_tok, out_tok),
        "model": model,
    }