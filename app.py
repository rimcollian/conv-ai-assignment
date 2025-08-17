import os, time, math
import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# -------------------------
# Config & Files
# -------------------------
st.set_page_config(page_title="RAG vs Fine-Tuned (Group 88)", layout="wide")

DATA_FILE = "data/data.txt"          # one chunk per line
QA_FILE   = "data/qa_pairs.csv"      # columns: question,answer
DEFAULT_CHUNK_SIZE = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Helpers
# -------------------------
def read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def load_qa(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["question","answer"])
    df = pd.read_csv(path)
    df = df.dropna(subset=["question","answer"])
    return df

def simple_guardrail(query: str) -> Tuple[bool, str]:
    bad = ["password", "hack", "malware", "exploit"]
    if any(w in query.lower() for w in bad):
        return False, "Query blocked by guardrail (unsafe terms)."
    # Out-of-scope detector: extremely generic geo questions
    oos = ["capital of", "weather", "population", "who is the president"]
    if any(p in query.lower() for p in oos):
        return False, "Out of scope for this app (financial QA only)."
    return True, ""

def tokenize_words(text: str) -> List[str]:
    return text.lower().split()

# -------------------------
# Cached resources (models)
# -------------------------
@st.cache_resource(show_spinner=True)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

@st.cache_resource(show_spinner=True)
def get_cross_encoder():
    # Light cross-encoder loaded directly with transformers (no datasets)
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE)
    mdl.eval()
    return tok, mdl

@st.cache_resource(show_spinner=True)
def get_generator():
    # Small, reliable generator for short answers
    gen_name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(gen_name)
    mdl = AutoModelForCausalLM.from_pretrained(gen_name).to(DEVICE)
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=0 if DEVICE=="cuda" else -1)
    return pipe

# -------------------------
# Build indexes (cached)
# -------------------------
@st.cache_data(show_spinner=True)
def build_indexes(chunk_size: int):
    texts = read_lines(DATA_FILE)
    if not texts:
        texts = [
            "Demo text: Please add your Reliance Industries Limited financial text lines to data/data.txt.",
            "Each line is treated as a chunk. Include Balance Sheet, P&L, cash flows, notes, etc.",
        ]

    # BM25
    tokenized = [tokenize_words(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    # Embeddings
    embedder = get_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # TF-IDF (for FT fallback and quick baselines)
    tfidf = TfidfVectorizer(min_df=1).fit(texts)
    X_tfidf = tfidf.transform(texts)

    metadata = {"tfidf": tfidf}
    return texts, bm25, embeddings, metadata

@st.cache_data(show_spinner=True)
def build_ft_index():
    qa = load_qa(QA_FILE)
    if qa.empty:
        return qa, None
    vec = TfidfVectorizer(min_df=1).fit(qa["question"])
    Xq = vec.transform(qa["question"])
    return qa, (vec, Xq)

# -------------------------
# Retrieval + Rerank
# -------------------------
def retrieve_hybrid(query: str,
                    texts: List[str],
                    bm25: BM25Okapi,
                    embeddings: np.ndarray,
                    top_bm25: int = 8,
                    top_dense: int = 8,
                    top_final: int = 5):
    # BM25 scores
    bm25_scores = bm25.get_scores(tokenize_words(query))
    bm25_idx = np.argsort(bm25_scores)[::-1][:top_bm25]

    # Dense scores
    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    dense_scores = cosine_similarity([q_emb], embeddings)[0]
    dense_idx = np.argsort(dense_scores)[::-1][:top_dense]

    # Union candidates
    cand = list(dict.fromkeys(list(bm25_idx) + list(dense_idx)))  # preserve order
    cand_texts = [texts[i] for i in cand]

    # Lightweight cross-encoder rerank
    tok, mdl = get_cross_encoder()
    pairs = [(query, t) for t in cand_texts]
    enc = tok([p[0] for p in pairs], [p[1] for p in pairs],
              padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits.squeeze(-1)  # shape: [N]
        scores = torch.sigmoid(logits).detach().cpu().numpy()

    # pick top_final
    order = np.argsort(scores)[::-1][:top_final]
    top = [(cand[idx], cand_texts[idx], float(scores[idx])) for idx in order]
    return top  # list of (idx, text, score in 0..1)

# -------------------------
# Answer generation (RAG)
# -------------------------
def generate_answer(contexts: List[str], query: str, max_new_tokens=80) -> str:
    pipe = get_generator()
    prompt = (
        "You are a helpful financial analyst. Using ONLY the context, answer concisely.\n\n"
        f"Context:\n{chr(10).join(f'- {c}' for c in contexts)}\n\n"
        f"Question: {query}\nAnswer:"
    )
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1)
    text = out[0]["generated_text"]
    ans = text.split("Answer:", 1)[-1].strip()
    return ans[:1000]

# -------------------------
# Fine-Tuned mode (hosted demo surrogate)
# -------------------------
def ft_answer(query: str, qa_df: pd.DataFrame, vec_pack, top_k=1):
    if qa_df is None or qa_df.empty or vec_pack is None:
        return "FT model not available (no qa_pairs.csv).", 0.0
    vec, Xq = vec_pack
    qv = vec.transform([query])
    sims = cosine_similarity(qv, Xq)[0]
    idx = int(np.argmax(sims))
    conf = float(sims[idx])
    return str(qa_df.iloc[idx]["answer"]), conf

# -------------------------
# UI
# -------------------------
st.title("Comparative Financial QA: RAG vs Fine-Tuned (Group 88)")
st.caption("Hybrid RAG uses BM25 + Embeddings + Cross-Encoder reranking (no datasets/pyarrow). FT uses a lightweight Q→A matcher for the hosted demo. The full fine-tuning is documented in your notebook.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["RAG (Hybrid)", "Fine-Tuned"])
    top_final = st.slider("Top passages (RAG)", 3, 10, 5)
    show_contexts = st.checkbox("Show retrieved contexts", value=True)
    st.markdown("---")
    st.subheader("Guardrails")
    enable_guardrail = st.checkbox("Enable input guardrail", value=True)
    st.markdown("---")
    st.subheader("Data")
    st.write("• Put your report chunks in `data/data.txt` (one chunk per line).")
    st.write("• Put ~50 Q/A pairs in `data/qa_pairs.csv` with columns `question,answer`.")
    st.markdown("---")
    st.caption("Device: " + DEVICE)

# Build indexes
with st.spinner("Building indexes..."):
    texts, bm25, embeddings, meta = build_indexes(DEFAULT_CHUNK_SIZE)
    qa_df, vec_pack = build_ft_index()

query = st.text_input("Ask a financial question about Reliance (last two years):", "")
go = st.button("Run")

if go and query.strip():
    ok, msg = simple_guardrail(query) if enable_guardrail else (True, "")
    if not ok:
        st.error(msg)
    else:
        if mode.startswith("RAG"):
            t0 = time.perf_counter()
            top = retrieve_hybrid(query, texts, bm25, embeddings, top_final=top_final)
            contexts = [t[1] for t in top]
            ce_scores = [t[2] for t in top]
            conf = float(np.mean(ce_scores)) if ce_scores else 0.0
            answer = generate_answer(contexts, query)
            elapsed = time.perf_counter() - t0

            st.subheader("Answer (RAG)")
            st.write(answer)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Confidence (avg CE)", f"{conf:.2f}")
            with c2: st.metric("Method", "RAG (Hybrid)")
            with c3: st.metric("Time (s)", f"{elapsed:.2f}")

            if show_contexts and len(contexts):
                st.markdown("**Top retrieved contexts:**")
                for i, (idx, txt, sc) in enumerate(top, 1):
                    st.write(f"{i}. *(score={sc:.2f})* {txt}")

        else:
            t0 = time.perf_counter()
            ans, conf = ft_answer(query, qa_df, vec_pack)
            elapsed = time.perf_counter() - t0

            st.subheader("Answer (Fine-Tuned demo)")
            st.write(ans)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Confidence (similarity)", f"{conf:.2f}")
            with c2: st.metric("Method", "Fine-Tuned (Q→A matcher)")
            with c3: st.metric("Time (s)", f"{elapsed:.2f}")

            if qa_df is None or qa_df.empty:
                st.info("Upload `data/qa_pairs.csv` with columns `question,answer` to enable FT demo.")
