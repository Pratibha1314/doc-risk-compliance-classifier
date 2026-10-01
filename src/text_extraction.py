import re
from pathlib import Path

import pdfplumber
from docx import Document


def extract_text(file, filename):
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        with pdfplumber.open(file) as pdf:
            return "\n\n".join((p.extract_text() or "") for p in pdf.pages)
    if ext == ".docx":
        return "\n\n".join(p.text for p in Document(file).paragraphs)
    return file.read().decode("utf-8", errors="ignore")


def clean_text(t):
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def split_into_clauses(text, min_words=8, max_words=200):
    text = text.replace("\r", "")
    # split on blank lines and on numbered headings like "1." / "2.1" / "Section 3" / "ARTICLE"
    parts = re.split(r"\n\s*\n|\n(?=\s*(?:\d+(?:\.\d+)*[.)]\s|Section\s+\d+|ARTICLE\s))", text)
    clauses = []
    for p in parts:
        if not p:
            continue
        p = re.sub(r"\s+", " ", p).strip()
        words = p.split()
        if len(words) > max_words:  # break very long chunks into sentence groups
            sentences = re.split(r"(?<=[.;])\s+", p)
            buf = []
            for s in sentences:
                buf.append(s)
                if len(" ".join(buf).split()) >= 100:
                    clauses.append(" ".join(buf))
                    buf = []
            if buf:
                clauses.append(" ".join(buf))
        else:
            clauses.append(p)
    return [c for c in clauses if len(c.split()) >= min_words]