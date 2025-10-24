# rag_module/core/ingest.py

import os
from typing import List, Dict
from rag_module.core.store import add_document

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


def read_pdf(file_path: str) -> List[str]:
    if not PdfReader:
        raise ImportError("Please install PyPDF2: pip install PyPDF2")
    reader = PdfReader(file_path)
    return [page.extract_text() for page in reader.pages if page.extract_text()]


def read_txt(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [text]


def chunk_text(text_list: List[str], chunk_size: int = 500) -> List[str]:
    chunks = []
    for text in text_list:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def process_file(file_path: str, metadata: Dict) -> str:
    file_path = os.path.expanduser(file_path)         # support ~/
    if not os.path.isabs(file_path):                  # make relative explicit
        file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text_list = read_pdf(file_path)
    elif ext == ".txt":                               # easy local test
        text_list = read_txt(file_path)
    else:
        raise ImportError(f"Unsupported file type: {ext}. Use .pdf or .txt for now.")

    chunks = chunk_text(text_list)
    doc_id = add_document(chunks, metadata)
    return doc_id
