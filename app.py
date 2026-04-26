import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# -------------------------------
# CONFIG
# -------------------------------
INDEX_FILE = "faiss_index.bin"
META_FILE = "metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
SIM_THRESHOLD = 0.25

# -------------------------------
# LOAD SYSTEM
# -------------------------------
@st.cache_resource
def load_system():
    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)

    embed_model = SentenceTransformer(EMBED_MODEL)

    return index, metadata, embed_model

# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve(query, index, metadata, model):
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, TOP_K)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "text": metadata[idx]["text"],
            "page": metadata[idx]["page"]
        })

    return results

# -------------------------------
# PROMPT
# -------------------------------
def build_prompt(question, chunks):
    context = "\n\n".join([
        f"(p.{c['page']}) {c['text'][:400]}"
        for c in chunks
    ])

    return f"""
You are a strict FYP handbook assistant.
Answer ONLY using the context.
Always mention page numbers.

Context:
{context}

Question:
{question}

Answer clearly:
"""

# -------------------------------
# STREAMLIT UI
# -------------------------------
def main():
    st.title("🔥 LLaMA 3 RAG Assistant (Ollama)")

    index, metadata, embed_model = load_system()

    query = st.text_input("Ask your question:")

    if st.button("Ask") and query:

        results = retrieve(query, index, metadata, embed_model)

        if results[0]["score"] < SIM_THRESHOLD:
            st.warning("Not found in handbook.")
            return

        prompt = build_prompt(query, results)

        # ✅ OLLAMA CALL (instead of transformers)
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response["message"]["content"]

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Sources"):
            for r in results:
                st.write(f"Page {r['page']} | Score {r['score']:.3f}")
                st.write(r["text"][:300])
                st.write("---")

if __name__ == "__main__":
    main()