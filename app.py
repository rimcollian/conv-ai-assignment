import streamlit as st
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os

# -------------------------
# Parameters
# -------------------------
chunk_size = 300
DATA_FILE = "data.txt"

# -------------------------
# Utility functions
# -------------------------
def load_data():
    if not os.path.exists(DATA_FILE):
        return ["This is a demo text. Add your own data.txt file."]
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

@st.cache_data(show_spinner=True)
def build_indexes(chunk_size):
    texts = load_data()

    # BM25
    tokenized_texts = [t.split(" ") for t in texts]
    bm25 = BM25Okapi(tokenized_texts)

    # Embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X = embedder.encode(texts, convert_to_tensor=False)

    # TF-IDF (optional)
    tfidf = TfidfVectorizer().fit(texts)
    X_tfidf = tfidf.transform(texts)

    meta = {"tfidf": tfidf}
    return texts, bm25, embedder, X, X_tfidf, meta

# -------------------------
# Search Function
# -------------------------
def search(query, texts, bm25, embedder, X, X_tfidf, meta, top_k=5):
    # BM25
    bm25_scores = bm25.get_scores(query.split(" "))

    # Embedding similarity
    query_emb = embedder.encode([query], convert_to_tensor=False)
    sim_scores = cosine_similarity(query_emb, X)[0]

    # Combine scores (simple sum)
    combined_scores = bm25_scores + sim_scores

    # Get top-k results
    top_idx = combined_scores.argsort()[::-1][:top_k]
    results = [(texts[i], combined_scores[i]) for i in top_idx]
    return results

# -------------------------
# Streamlit UI
# -------------------------
st.title("Conversational AI Assignment")
st.write("Search your knowledge base using BM25 + Embeddings (no CrossEncoder).")

texts, bm25, embedder, X, X_tfidf, meta = build_indexes(chunk_size)

query = st.text_input("Enter your query:")
if query:
    results = search(query, texts, bm25, embedder, X, X_tfidf, meta, top_k=5)
    st.subheader("Results:")
    for r, score in results:
        st.write(f"**Score:** {score:.2f} — {r}")
