# ingest.py
"""
Assignment Part Covered:
1. Load & chunk (with page numbers)
2. Embed & index
"""

import os
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

hf_token = os.getenv("HUGGING_FACE_TOKEN")
print("Token loaded: ", hf_token is not None)

login(hf_token)

# -------------------------------
# CONFIG
# -------------------------------
PDF_PATH = "FYP-Handbook\\3. FYP-Handbook-2023.pdf"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
META_FILE = "metadata.pkl"

# -------------------------------
# LOGGING SETUP
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="🔹 %(message)s"
)

# -------------------------------
# STEP 1: LOAD PDF
# -------------------------------
def load_pdf(pdf_path):
    logging.info("Loading PDF...")

    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            pages.append({
                "page": i + 1,
                "text": text
            })

            logging.info(f"Loaded page {i + 1} | chars: {len(text)}")

    logging.info(f"Total pages loaded: {len(pages)}")
    return pages

# -------------------------------
# STEP 2: CHUNKING
# -------------------------------
def chunk_text(pages):
    logging.info("Starting chunking process...")

    # STEP 1: merge all text first
    full_text = "\n".join([p["text"] for p in pages])

    words = full_text.split()

    chunks = []

    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_words = words[i:i + CHUNK_SIZE]

        chunks.append({
            "text": " ".join(chunk_words),
            "page": "merged"
        })

    logging.info(f"Total chunks created: {len(chunks)}")

    logging.info("Sample chunk preview:")
    logging.info(chunks[0]["text"][:300])

    return chunks

# -------------------------------
# STEP 3: EMBEDDINGS
# -------------------------------
def create_embeddings(chunks, model):
    logging.info("Loading embedding model...")
    texts = [c["text"] for c in chunks]

    logging.info("Creating embeddings (this may take time)...")
    embeddings = model.encode(texts, show_progress_bar=True)

    logging.info(f"Embeddings shape: {embeddings.shape}")
    return np.array(embeddings)

# -------------------------------
# STEP 4: BUILD FAISS INDEX
# -------------------------------
def build_index(embeddings):
    logging.info("Building FAISS index...")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    logging.info(f"Index built with {index.ntotal} vectors")
    return index

# -------------------------------
# SAVE DATA
# -------------------------------
def save_data(index, chunks):
    logging.info("Saving index and metadata...")

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    logging.info("Files saved successfully!")

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    logging.info("Starting ingestion pipeline")

    pages = load_pdf(PDF_PATH)
    chunks = chunk_text(pages)

    model = SentenceTransformer(EMBED_MODEL)

    embeddings = create_embeddings(chunks, model)
    index = build_index(embeddings)

    save_data(index, chunks)

    logging.info("INGESTION COMPLETE")

# -------------------------------
if __name__ == "__main__":
    main()