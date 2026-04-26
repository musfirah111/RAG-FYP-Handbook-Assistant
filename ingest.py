import os
import pickle
import logging
import numpy as np
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from huggingface_hub import login

# -------------------------------
# INIT
# -------------------------------
load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

hf_token = os.getenv("HUGGING_FACE_TOKEN")
if hf_token:
    login(hf_token)

# -------------------------------
# CONFIG
# -------------------------------
PDF_PATH = r"FYP-Handbook\3. FYP-Handbook-2023.pdf"
CHUNK_SIZE = 250          # FIXED (was too large)
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"

INDEX_FILE = "faiss_index.bin"
META_FILE = "metadata.pkl"

# -------------------------------
# LOGGING
# -------------------------------
logging.basicConfig(level=logging.INFO, format="🔹 %(message)s")

# -------------------------------
# LOAD PDF + OCR FALLBACK
# -------------------------------
def load_pdf(pdf_path):
    logging.info("Loading PDF + OCR fallback...")

    pages = []
    images = convert_from_path(pdf_path, dpi=300)
    reader = PdfReader(pdf_path)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if not text or len(text.strip()) < 100:
            text = pytesseract.image_to_string(images[i], config="--oem 3 --psm 6")

        if text and text.strip():
            pages.append({
                "page": i + 1,
                "text": text.strip()
            })

    logging.info(f"Pages loaded: {len(pages)}")
    return pages

# -------------------------------
# CLEAN TEXT (IMPORTANT FIX)
# -------------------------------
def clean_text(text):
    text = " ".join(text.split())
    return text

# -------------------------------
# CHUNKING (FIXED VERSION)
# -------------------------------
def chunk_text(pages):
    logging.info("Chunking text properly...")

    chunks = []

    for page in pages:
        words = clean_text(page["text"]).split()

        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = words[i:i + CHUNK_SIZE]

            chunks.append({
                "text": " ".join(chunk),
                "page": page["page"]
            })

    logging.info(f"Total chunks: {len(chunks)}")
    return chunks

# -------------------------------
# EMBEDDINGS
# -------------------------------
def create_embeddings(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)

# -------------------------------
# FAISS INDEX
# -------------------------------
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index

# -------------------------------
# SAVE
# -------------------------------
def save(index, chunks):
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

# -------------------------------
# MAIN
# -------------------------------
def main():
    pages = load_pdf(PDF_PATH)

    chunks = chunk_text(pages)

    model = SentenceTransformer(EMBED_MODEL)

    embeddings = create_embeddings(chunks, model)

    index = build_index(embeddings)

    save(index, chunks)

    logging.info("DONE 🚀")

if __name__ == "__main__":
    main()