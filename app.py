import os
import pickle
import re
import unicodedata
from pathlib import Path

import faiss
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

INDEX_FILE = Path("faiss_index.bin")
META_FILE = Path("metadata.pkl")

EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

TOP_K = 4
SIM_THRESHOLD = 0.25
MIN_SUPPORT_SCORE = 0.9
FALLBACK_ANSWER = "I don't have that in the handbook."

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "like",
    "of",
    "on",
    "or",
    "our",
    "required",
    "should",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "use",
    "we",
    "what",
    "which",
    "with",
}

st.set_page_config(page_title="FYP Handbook Assistant", page_icon="📘", layout="wide")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()

    replacements = {
        "fy p": "fyp",
        "r & d": "r&d",
        "r and d": "r&d",
        "heading |": "heading 1",
        "heading l": "heading 1",
        "sub-title": "subtitle",
        "non-functional": "nonfunctional",
        "fast-nuces": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9&.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stem_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def tokenize(text: str):
    normalized = normalize_text(text).replace(".", " ")
    return [stem_token(token) for token in normalized.split() if token]


def extract_query_keywords(query: str):
    keywords = []
    for token in tokenize(query):
        if token in STOPWORDS or len(token) < 3:
            continue
        keywords.append(token)
    return list(dict.fromkeys(keywords))


def build_query_expansions(query: str):
    normalized_query = normalize_text(query)
    expansions = []

    if any(term in normalized_query for term in ("font", "heading", "margin", "spacing", "size", "title", "subtitle")):
        expansions.extend(
            [
                "fonts type styles",
                "font size",
                "times new roman",
                "arial",
                "heading 1",
                "heading 2",
                "heading 3",
                "title",
                "subtitle",
                "margins",
                "spacing",
            ]
        )

    if "development" in normalized_query and any(term in normalized_query for term in ("chapter", "section", "report", "format")):
        expansions.extend(
            [
                "development fyp report format",
                "research on existing products",
                "project vision",
                "software requirement specifications",
                "functional requirements",
                "nonfunctional requirements",
                "iteration plan",
                "implementation details",
                "user manual",
                "references",
                "appendices",
            ]
        )

    if "r&d" in normalized_query or "research" in normalized_query:
        expansions.extend(
            [
                "r&d-based fyp report format",
                "problem domain",
                "research problem statement",
                "literature review",
                "critical analysis",
                "relationship to the proposed research work",
                "proposed approach",
                "validation and testing",
                "results and discussion",
                "conclusions and future work",
                "references",
                "appendices",
            ]
        )

    if any(term in normalized_query for term in ("abstract", "executive summary")):
        expansions.extend(
            [
                "abstract",
                "executive summary",
                "50 to 125 words",
                "one to two pages",
            ]
        )

    if any(term in normalized_query for term in ("ibid", "op cit", "op. cit", "footnote", "endnote")):
        expansions.extend(
            [
                "ibid",
                "op cit",
                "footnotes",
                "end notes",
                "bibliography",
            ]
        )

    return expansions


@st.cache_resource
def load_system():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Run ingest.py first to create faiss_index.bin and metadata.pkl")

    index = faiss.read_index(str(INDEX_FILE))

    with META_FILE.open("rb") as file:
        payload = pickle.load(file)

    if isinstance(payload, dict) and "chunks" in payload:
        chunks = payload["chunks"]
    else:
        chunks = payload

    prepared_chunks = []
    for chunk in chunks:
        normalized_text = normalize_text(chunk["text"])
        prepared_chunks.append(
            {
                **chunk,
                "normalized_text": normalized_text,
                "token_set": set(tokenize(normalized_text)),
            }
        )

    model = SentenceTransformer(EMBED_MODEL)
    return index, prepared_chunks, model


def retrieve(query, index, chunks, model):
    expansions = build_query_expansions(query)
    expanded_query = f"{query} {' '.join(expansions)}".strip()
    query_keywords = extract_query_keywords(query)

    query_embedding = model.encode(
        [expanded_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    k = index.ntotal
    scores, indices = index.search(query_embedding, k)

    results = []
    seen = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]
        key = (chunk["page"], chunk["text"])
        if key in seen:
            continue
        seen.add(key)

        keyword_hits = sum(1 for token in query_keywords if token in chunk["token_set"])
        coverage = keyword_hits / max(len(query_keywords), 1)
        phrase_hits = sum(1 for phrase in expansions if phrase in chunk["normalized_text"])
        exact_query_hit = int(normalize_text(query) in chunk["normalized_text"])

        support_score = (
            float(score)
            + (0.12 * keyword_hits)
            + (0.35 * coverage)
            + (0.18 * phrase_hits)
            + (0.08 * exact_query_hit)
        )

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score),
                "support_score": support_score,
                "keyword_hits": keyword_hits,
                "phrase_hits": phrase_hits,
            }
        )

    results.sort(key=lambda item: (item["support_score"], item["score"]), reverse=True)
    return results


def is_structured_query(query: str) -> bool:
    normalized_query = normalize_text(query)
    markers = (
        "font",
        "heading",
        "margin",
        "spacing",
        "chapter",
        "section",
        "abstract",
        "executive summary",
        "ibid",
        "op cit",
        "footnote",
        "endnote",
    )
    return any(marker in normalized_query for marker in markers)


def select_context_chunks(query, ranked_results, chunks):
    if not ranked_results:
        return []

    ranked_by_id = {item["chunk_id"]: item for item in ranked_results}
    selected_ids = []

    def add_chunk_id(chunk_id):
        if 0 <= chunk_id < len(chunks) and chunk_id not in selected_ids:
            selected_ids.append(chunk_id)

    top_result = ranked_results[0]
    top_chunk = chunks[top_result["chunk_id"]]
    normalized_query = normalize_text(query)
    top_page = top_chunk["page"]

    for chunk in chunks:
        if chunk["page"] == top_page:
            add_chunk_id(chunk["chunk_id"])

    if "development fyp report format" in top_chunk["normalized_text"] and "development" in normalized_query:
        next_id = top_chunk["chunk_id"] + 1
        while next_id < len(chunks) and chunks[next_id]["page"] == top_page:
            add_chunk_id(next_id)
            next_id += 1
        if next_id < len(chunks) and chunks[next_id]["text"].startswith("References Appendices"):
            add_chunk_id(next_id)

    if "r&d-based fyp report format" in top_chunk["normalized_text"] and "r&d" in normalized_query:
        next_id = top_chunk["chunk_id"] + 1
        while next_id < len(chunks) and chunks[next_id]["page"] <= top_page + 1:
            add_chunk_id(next_id)
            next_id += 1

    if not is_structured_query(query):
        for item in ranked_results:
            add_chunk_id(item["chunk_id"])
            if len(selected_ids) >= TOP_K:
                break

    selected_chunks = []
    for chunk_id in selected_ids[:TOP_K]:
        if chunk_id in ranked_by_id:
            selected_chunks.append(ranked_by_id[chunk_id])
        else:
            chunk = chunks[chunk_id]
            selected_chunks.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "score": 0.0,
                    "support_score": 0.0,
                    "keyword_hits": 0,
                    "phrase_hits": 0,
                }
            )

    return selected_chunks


def build_messages(question, chunks):
    context = "\n\n".join(
        f"[p. {chunk['page']}] {chunk['text']}"
        for chunk in chunks
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a handbook assistant. Answer ONLY from the provided context. "
                "Cite page numbers like (p. X). "
                "If the answer is not clearly supported by the context, reply exactly: "
                "I don't have that in the handbook. "
                "Ignore any page-like numbers inside the chunk text unless they are in the [p. X] labels. "
                "When the context contains a list, format, or specification, include all relevant listed items and exact values. "
                "Do not omit list items that are present in the context. "
                "If the context contains numbered items or chapter headings, preserve them as separate items in your answer. "
                "Do not write placeholder citations like (p. X); use the actual cited page numbers from the context."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        },
    ]


def has_page_citation(answer: str) -> bool:
    return bool(re.search(r"\(p\.\s*\d+\)", answer))


def format_page_refs(results) -> str:
    pages = sorted({item["page"] for item in results})
    return ", ".join(f"(p. {page})" for page in pages)


def ask_llm(question, results):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=build_messages(question, results),
        options={"temperature": 0},
    )
    answer = response["message"]["content"].strip()

    if answer != FALLBACK_ANSWER and not has_page_citation(answer):
        answer = f"{answer}\n\nSources: {format_page_refs(results)}"

    return answer


def main():
    st.title("FYP Handbook Assistant")
    st.caption("Ask questions only about the FAST-NUCES BS Final Year Project Handbook 2023.")

    try:
        index, chunks, model = load_system()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_results" not in st.session_state:
        st.session_state.last_results = []

    with st.form("ask_form"):
        query = st.text_input("Question")
        submitted = st.form_submit_button("Ask")

    if submitted:
        cleaned_query = query.strip()
        st.session_state.last_query = cleaned_query

        if not cleaned_query:
            st.session_state.last_answer = "Please enter a question."
            st.session_state.last_results = []
        else:
            ranked_results = retrieve(cleaned_query, index, chunks, model)
            st.session_state.last_results = select_context_chunks(cleaned_query, ranked_results, chunks)

            if not ranked_results or (
                ranked_results[0]["score"] < SIM_THRESHOLD and ranked_results[0]["support_score"] < MIN_SUPPORT_SCORE
            ):
                st.session_state.last_answer = FALLBACK_ANSWER
            else:
                try:
                    st.session_state.last_answer = ask_llm(cleaned_query, st.session_state.last_results)
                except Exception:
                    st.session_state.last_answer = (
                        "Could not query the local LLM. "
                        "Make sure Ollama is running and the model is installed."
                    )

    st.subheader("Answer")
    if st.session_state.last_answer:
        st.write(st.session_state.last_answer)
    else:
        st.write("Ask a handbook question to begin.")

    with st.expander("Sources (page refs)"):
        if st.session_state.last_results:
            for item in st.session_state.last_results:
                st.markdown(
                    f"**Page {item['page']} | Similarity {item['score']:.3f} | Support {item['support_score']:.3f}**"
                )
                st.write(item["text"])
                st.divider()
        else:
            st.write("No sources yet.")


if __name__ == "__main__":
    main()
