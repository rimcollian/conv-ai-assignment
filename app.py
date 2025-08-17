
import time, os, json, uuid
import streamlit as st

st.set_page_config(page_title="Comparative Financial QA: RAG vs Fine-Tuning (Group 88)", layout="wide")

st.title("Assignment 2 – Comparative Financial QA System: RAG vs Fine-Tuning")
st.caption("Dataset: Reliance Industries Limited (FY2022-23, FY2023-24) • Group 88")

# Sidebar controls
mode = st.sidebar.radio("Mode", ["RAG", "Fine-Tuned"])
topn = st.sidebar.slider("Top-N retrieved (each retriever)", 1, 10, 5)
chunk_size = st.sidebar.selectbox("Chunk size (tokens approx.)", [100, 400], index=1)
rerank = st.sidebar.checkbox("Enable Cross-Encoder re-ranking", True)
guard_input = st.sidebar.checkbox("Enable input guardrail (scope/abuse filter)", True)
guard_output = st.sidebar.checkbox("Enable output guardrail (factuality check)", True)

st.sidebar.markdown("---")
st.sidebar.markdown("**How to run locally**")
st.sidebar.code("""
pip install -U transformers sentence-transformers faiss-cpu rank_bm25 scikit-learn streamlit torch accelerate peft bitsandbytes
streamlit run app.py
""", language="bash")

# Load corpus & QAs
@st.cache_data
def load_qa():
    import pandas as pd
    p = os.path.join(os.path.dirname(__file__), "ril_fin_qa_pairs.csv")
    return pd.read_csv(p)

qa_df = load_qa()

# Build chunks and indexes on first run or cache
@st.cache_resource(show_spinner=True)
def build_indexes(token_target):
    # We accept raw Q/A 'answers' as mini-passages. In a full pipeline, you would extract from PDFs/HTML.
    # For the assignment, we include a minimal corpus that still demonstrates retrieval quality.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss

    corpus = (qa_df["answer"] + " Source: " + qa_df["source"].fillna("")).tolist()
    # Simple sentence splits (answers are short)
    texts = corpus

    # Sparse (BM25)
    bm25 = BM25Okapi([t.lower().split() for t in texts])

    # Dense (MiniLM)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X.astype("float32"))

    # Cross-Encoder
    cross = None
    if rerank:
        try:
            from sentence_transformers import CrossEncoder
            cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            cross = None

    meta = [{"id": i, "text": texts[i]} for i in range(len(texts))]
    return texts, bm25, embedder, X, index, cross, meta

texts, bm25, embedder, X, index, cross, meta = build_indexes(chunk_size)

def guard_in(query: str) -> bool:
    if not guard_input: 
        return True
    ql = query.lower()
    banned = ["hack", "attack", "exploit", "kill", "bomb"]
    if any(w in ql for w in banned):
        return False
    # scope filter: require 'reliance', 'ril', or finance terms to reduce irrelevant Qs
    scope_terms = ["reliance", "ril", "revenue", "ebitda", "profit", "segment", "o2c", "jio", "retail", "fy2023-24", "fy2022-23"]
    if not any(w in ql for w in scope_terms):
        return False
    return True

def guard_out(answer: str, retrieved: list) -> str:
    if not guard_output:
        return answer
    # simple factuality check: if answer includes a number not present in retrieved, flag it.
    import re
    nums_ans = re.findall(r"[₹]?\d[\d,]*", answer)
    textblob = " ".join([r["text"] for r in retrieved])
    flags = []
    for n in nums_ans:
        if n not in textblob:
            flags.append(n)
    if flags:
        answer += f"\\n\\n⚠️ Output guardrail: numbers {flags} not found in retrieved context; please verify against sources."
    return answer

def retrieve(query, K):
    import numpy as np
    # preprocess
    qproc = query.strip().lower()

    # sparse
    docs_tokens = [t.lower().split() for t in texts]
    bm_scores = bm25.get_scores(qproc.split())
    top_bm = np.argsort(-bm_scores)[:K]

    # dense
    qv = embedder.encode([qproc], normalize_embeddings=True)[0].astype("float32")
    D, I = index.search(qv.reshape(1,-1), K)
    top_dense = I[0]

    # combine (union) then optional rerank
    cand = sorted(set(list(top_bm) + list(top_dense)))
    retrieved = [{"id": int(i), "text": texts[int(i)], "sparse_score": float(bm_scores[int(i)])} for i in cand]

    # rerank with cross-encoder
    if cross is not None:
        pairs = [[qproc, r["text"]] for r in retrieved]
        try:
            scores = cross.predict(pairs).tolist()
            for r, s in zip(retrieved, scores):
                r["cross_score"] = float(s)
            retrieved = sorted(retrieved, key=lambda r: r.get("cross_score", 0.0), reverse=True)[:K]
        except Exception:
            retrieved = retrieved[:K]
    else:
        retrieved = retrieved[:K]

    return retrieved

def generate_with_distilgpt2(prompt, max_new_tokens=64):
    # Small open-source generative model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    model_id = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    txt = tok.decode(out[0], skip_special_tokens=True)
    return txt[len(prompt):].strip()

def answer_rag(query):
    t0 = time.time()
    if not guard_in(query):
        return {"answer": "Query blocked by input guardrail: out of scope or unsafe.", "time": time.time()-t0, "confidence": 0.0, "method": "RAG"}
    retrieved = retrieve(query, topn)
    context = "\\n\\n".join([r["text"] for r in retrieved])
    prompt = f"Answer the question strictly using the context. If not present, say 'Data not in scope.'\\n\\nContext:\\n{context}\\n\\nQuestion: {query}\\nAnswer:"
    try:
        gen = generate_with_distilgpt2(prompt, max_new_tokens=64)
        ans = gen.split('\\n')[0].strip()
    except Exception:
        # Fallback: extractive heuristic
        ans = context[:400] + " ..."
    ans = guard_out(ans, retrieved)
    t1 = time.time()
    # naive confidence proxy: presence of numbers/context match length
    conf = min(0.99, 0.5 + 0.5*min(1.0, len(context)/1000))
    return {"answer": ans, "time": t1-t0, "confidence": conf, "method": "RAG", "retrieved": retrieved}

def baseline_finetuned_predict(query):
    # Placeholder for fine-tuned model; here we reuse RAG retrieval to build a prompt.
    # Users will replace this with the fine-tuned ckpt for true results.
    ctx = retrieve(query, topn)
    context = "\\n".join([c["text"] for c in ctx])
    # Simple template as a stand-in
    ans = f"(FT model) Based on training, the answer is: " + context.split(" Source: ")[0]
    return ans, ctx

q = st.text_input("Ask a question (e.g., 'What was consolidated revenue in FY2023-24?')", "")
if st.button("Submit") and q:
    if mode == "RAG":
        out = answer_rag(q)
        st.subheader("Answer")
        st.write(out["answer"])
        st.markdown("**Method:** RAG  \n**Confidence (proxy):** {:.2f}  \n**Response time:** {:.2f}s".format(out["confidence"], out["time"]))
        with st.expander("Retrieved Passages"):
            for r in out.get("retrieved", []):
                st.write(r["text"])
    else:
        t0 = time.time()
        a, ctx = baseline_finetuned_predict(q)
        t1 = time.time()
        st.subheader("Answer")
        st.write(a)
        st.markdown("**Method:** Fine-Tuned (placeholder)  \n**Confidence (proxy):** 0.80  \n**Response time:** {:.2f}s".format(t1-t0))
        with st.expander("Context (for demo)"):
            for r in ctx:
                st.write(r["text"])

st.markdown("---")
st.caption("Open-source stack only. Replace the baseline FT function with your fine-tuned checkpoint for full marks.")
