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
TOP_K = 8
SIM_THRESHOLD = 0.10

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="FYP Assistant",
    layout="wide"
)

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
        text = metadata[idx]["text"]

        # keyword boost (fix font vs fonts issue)
        bonus = 0
        for word in query.lower().split():
            if word in text.lower():
                bonus += 0.08

        results.append({
            "score": float(score + bonus),
            "text": text,
            "page": metadata[idx]["page"]
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

# -------------------------------
# PROMPT (FIXED + STRICT)
# -------------------------------
def build_prompt(question, chunks):

    context = "\n\n".join([
        f"(Page {c['page']}) {c['text']}"
        for c in chunks
    ])

    return f"""
You are a STRICT EXAM-GRADE EXTRACTION SYSTEM.

RULES (MUST FOLLOW):
1. Extract EXACT values (numbers, limits, formats)
2. NEVER paraphrase numbers
3. NEVER say "not mentioned" if any hint exists
4. If multiple rules exist → list ALL
5. ALWAYS include page numbers
6. Output ONLY bullet points

Context:
{context}

Question:
{question}

FINAL ANSWER:
"""

# -------------------------------
# SESSION STATE
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

results = []  # prevent undefined error

# -------------------------------
# UI
# -------------------------------
def main():
    global results

    st.title("FYP RAG Assistant (LLaMA 3)")

    index, metadata, embed_model = load_system()

    query = st.chat_input("Ask about FYP handbook...")

    if query:
        st.session_state.history.append(("user", query))

        results = retrieve(query, index, metadata, embed_model)

        if results[0]["score"] < SIM_THRESHOLD:
            answer = "Not found in handbook."
        else:
            prompt = build_prompt(query, results)

            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["message"]["content"]

        st.session_state.history.append(("assistant", answer))

    # -------------------------------
    # CHAT DISPLAY
    # -------------------------------
    for role, msg in st.session_state.history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

    # -------------------------------
    # DEBUG PANEL
    # -------------------------------
    if query and results:
        with st.expander("Retrieved Context (Debug)"):
            for r in results:
                st.markdown(f"**Page {r['page']} | Score {r['score']:.3f}**")
                st.write(r["text"])
                st.divider()

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()