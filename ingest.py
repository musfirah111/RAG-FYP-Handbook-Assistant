import os
import logging
import pickle
import re
import unicodedata
from pathlib import Path

import faiss
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("FYP-Handbook") / "3. FYP-Handbook-2023.pdf"
INDEX_FILE = Path("faiss_index.bin")
META_FILE = Path("metadata.pkl")

CHUNK_SIZE = 120
CHUNK_OVERLAP = 25
MIN_CHUNK_WORDS = 30
EMBED_MODEL = "all-MiniLM-L6-v2"
OCR_DPI = 300

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
elif DEFAULT_TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(DEFAULT_TESSERACT_PATH)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")

    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "\u00ad": "",
        "\xa0": " ",
        "FY P": "FYP",
        "Y ear": "Year",
        "Pr oject": "Project",
        "T imes": "Times",
        "Literatur e": "Literature",
        "Sub-title": "Subtitle",
        "sub-title": "subtitle",
        "non-functional": "nonfunctional",
        "Heading |": "Heading 1",
        "Heading l": "Heading 1",
        "1 1.69": "11.69",
        "References Appendices R&D-Based FYP Report Format": "References. Appendices. R&D-Based FYP Report Format",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"FAST-NUCES\s*\d+\s*BS Final Year Project Handbook 2023",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d+\s+FAST-NUCES\b(?:\s+\d+)?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bFAST-NUCES\b\s*\d*", " ", text, flags=re.IGNORECASE)
    text = text.replace("www.nu.edu.pk", " ")

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if "BS Final Year Project Handbook 2023" in line:
            continue
        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pdf_pages(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logging.info("Loading PDF pages with OCR from %s", pdf_path)
    images = convert_from_path(str(pdf_path), dpi=OCR_DPI)
    pages = []

    for page_num, image in enumerate(images, start=1):
        raw_text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
        text = clean_text(raw_text)

        if text:
            pages.append({"page": page_num, "text": text})
        else:
            logging.warning("Skipped OCR-empty page %s", page_num)

    logging.info("Pages loaded: %s", len(pages))
    return pages


def chunk_pages(pages):
    logging.info("Chunking pages")
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    if step <= 0:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        page_chunks = []
        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_SIZE]
            if not chunk_words:
                continue

            if len(chunk_words) < MIN_CHUNK_WORDS and page_chunks:
                page_chunks[-1]["text"] += " " + " ".join(chunk_words)
            else:
                page_chunks.append(
                    {
                        "page": page["page"],
                        "text": " ".join(chunk_words),
                    }
                )

        for chunk in page_chunks:
            chunk["chunk_id"] = len(chunks)
            chunks.append(chunk)

    logging.info("Total chunks created: %s", len(chunks))
    return chunks


def create_embeddings(chunks, model_name: str):
    logging.info("Creating embeddings with %s", model_name)
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    logging.info("Building FAISS index")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_artifacts(index, chunks):
    faiss.write_index(index, str(INDEX_FILE))

    payload = {
        "chunks": chunks,
        "config": {
            "pdf_path": str(PDF_PATH),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_words": MIN_CHUNK_WORDS,
            "embed_model": EMBED_MODEL,
        },
    }

    with META_FILE.open("wb") as file:
        pickle.dump(payload, file)

    logging.info("Saved index to %s", INDEX_FILE)
    logging.info("Saved metadata to %s", META_FILE)


def main():
    pages = load_pdf_pages(PDF_PATH)
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No chunks were created from the PDF")

    embeddings = create_embeddings(chunks, EMBED_MODEL)
    index = build_faiss_index(embeddings)
    save_artifacts(index, chunks)

    logging.info("Ingestion complete")


if __name__ == "__main__":
    main()
