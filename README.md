# FYP Handbook Assistant (RAG)

A simple Retrieval-Augmented Generation (RAG) system that answers questions from the **FYP Handbook 2023** using OCR, FAISS, and Llama 3.

## Requirements

- Python 3.10+
- Tesseract OCR
- Ollama
- Llama 3 model

## Python Libraries

Install the required packages:

```bash
pip install pdf2image pytesseract sentence-transformers faiss-cpu ollama pillow numpy
```

## Setup

1. Install **Tesseract OCR**.
2. Install **Ollama** and pull the Llama 3 model:

```bash
ollama pull llama3
```

3. Run the project.

```bash
python main.py
```