import os

import joblib
import streamlit as st

from llm_utils import classify, get_client, get_model

st.set_page_config(page_title="Sentiment: Classic vs LLM", page_icon="🎬")

LABELS = {1: "Positive 😊", 0: "Negative 😞", -1: "Unclear ❓"}
NAMES = {
    "classic": "TF-IDF + Logistic Regression",
    "zero_shot": "LLM zero-shot",
    "few_shot": "LLM few-shot",
    "cot": "LLM chain-of-thought",
}
FREE_CALLS = 10  # per visitor session when using the owner's key


@st.cache_resource
def load_model():
    return joblib.load("models/tfidf_lr.joblib")


model = load_model()

if "llm_calls" not in st.session_state:
    st.session_state.llm_calls = 0

owner_key = None
try:
    owner_key = st.secrets["GROQ_API_KEY"]
except Exception:
    owner_key = os.getenv("GROQ_API_KEY")

st.title("🎬 Sentiment Analysis: Classic ML vs Prompted LLM")
st.write(
    "Type a movie review and compare a traditional **TF-IDF + Logistic Regression** "
    "model against an LLM using **zero-shot**, **few-shot**, and **chain-of-thought** prompts."
)

with st.sidebar:
    st.header("Settings")
    user_key = st.text_input("Your Groq API key (optional)", type="password")
    st.caption(
        "Your key is used only for this session and is not stored. "
        f"Without your own key, you get {FREE_CALLS} free LLM calls per session."
    )
    st.markdown("---")
    st.caption("Dataset: IMDB 50k reviews.")

default_text = "The plot was predictable, but the performances were so good that I didn't care."
text = st.text_area("Movie review", value=default_text, height=150)

choice = st.selectbox(
    "Method",
    ["Compare all"] + list(NAMES.keys()),
    format_func=lambda k: k if k == "Compare all" else NAMES[k],
)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter a review.")
        st.stop()

    methods = list(NAMES.keys()) if choice == "Compare all" else [choice]
    llm_needed = [m for m in methods if m != "classic"]

    api_key = user_key or None
    using_owner_key = False
    if llm_needed and not api_key:
        if owner_key:
            api_key = owner_key
            using_owner_key = True
        else:
            st.error("LLM methods need an API key. Paste yours in the sidebar.")
            st.stop()

    if using_owner_key and st.session_state.llm_calls + len(llm_needed) > FREE_CALLS:
        st.error("Free call limit reached for this session. Paste your own API key in the sidebar.")
        st.stop()

    try:
        client = get_client(api_key) if llm_needed else None
        if client:
            st.caption(f"LLM model: {get_model(client)}")
    except Exception as e:
        st.error(f"Could not start the LLM client: {e}")
        st.stop()

    for m in methods:
        st.subheader(NAMES[m])
        if m == "classic":
            proba = model.predict_proba([text])[0]
            pred = int(proba.argmax())
            st.write(f"**Prediction:** {LABELS[pred]}")
            st.write(f"Confidence: {proba.max():.1%}")
            st.caption("Cost: $0 | Latency: a few milliseconds")
        else:
            with st.spinner("Asking the LLM..."):
                try:
                    r = classify(text, m, client=client)
                except Exception as e:
                    st.error(f"API error: {e}")
                    continue
            if using_owner_key:
                st.session_state.llm_calls += 1
            st.write(f"**Prediction:** {LABELS[r['pred']]}")
            st.caption(
                f"Latency: {r['latency_s']:.2f}s | "
                f"Tokens: {r['in_tokens']} in / {r['out_tokens']} out"
            )
            if m == "cot":
                with st.expander("See the model's reasoning"):
                    st.write(r["raw"])