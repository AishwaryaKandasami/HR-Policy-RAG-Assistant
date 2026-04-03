"""
hr_doc_loader.py — HR Document Parser
=======================================
Parses PDF, DOCX, and TXT/MD files into structured records with metadata.

Each record produced:
  {
    "page_content": str,           # raw text of this page/section
    "metadata": {
      "doc_title":       str,      # human-readable title (from filename)
      "doc_type":        str,      # handbook | policy | procedure | guide | document
      "department":      str,      # "All" by default; overridable
      "section_heading": str,      # nearest heading above the text
      "page_number":     int,      # page (PDF) or section index (DOCX)
      "source_filename": str,      # original uploaded filename
      "ingested_at":     str,      # ISO-8601 UTC timestamp
    }
  }
"""

import pathlib
import re
from datetime import datetime, timezone

import pdfplumber
from docx import Document as DocxDocument


# ── Metadata helpers ───────────────────────────────────────────────

def _infer_doc_type(filename: str) -> str:
    """Infer document type from filename keywords."""
    name = filename.lower()
    if "handbook" in name:
        return "handbook"
    if "policy" in name or "policies" in name:
        return "policy"
    if "procedure" in name:
        return "procedure"
    if "guide" in name or "guidance" in name:
        return "guide"
    if "template" in name:
        return "template"
    if "circular" in name:
        return "circular"
    if "code" in name and "conduct" in name:
        return "policy"
    return "document"


def _infer_doc_title(filename: str) -> str:
    """Create a human-readable title from filename stem."""
    stem = pathlib.Path(filename).stem
    # Strip common prefixes like acas_, cipd_, gov_
    stem = re.sub(r"^(acas|cipd|gov|hr)_?", "", stem, flags=re.IGNORECASE)
    # Replace underscores/hyphens with spaces and title-case
    title = re.sub(r"[_\-]+", " ", stem).strip().title()
    return title or filename


def _is_heading_line(line: str) -> bool:
    """
    Heuristic: treat a line as a section heading if it is:
    - Short (< 80 chars)
    - Starts with a number+dot (e.g. "3. Long-Term Absence")
    - OR is all-uppercase with no trailing punctuation typical of body text
    - OR ends with ':' and is short
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if re.match(r"^\d+[\.\)]\s+\w", stripped):
        return True
    if stripped.isupper() and len(stripped.split()) <= 8:
        return True
    if stripped.endswith(":") and len(stripped) < 60:
        return True
    return False


# ── Parsers ────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a text-based PDF, page by page.
    Tracks the most recent heading seen to attach as section_heading metadata.
    """
    records = []
    filename = pathlib.Path(file_path).name
    doc_title = _infer_doc_title(filename)
    doc_type = _infer_doc_type(filename)
    ingested_at = datetime.now(timezone.utc).isoformat()
    current_heading = "General"

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or len(text.strip()) < 40:
                continue  # skip blank / header-only pages

            # Update current heading from page text
            for line in text.split("\n"):
                if _is_heading_line(line):
                    current_heading = line.strip().rstrip(":").strip()
                    break  # use first heading found on the page

            records.append({
                "page_content": text.strip(),
                "metadata": {
                    "doc_title":       doc_title,
                    "doc_type":        doc_type,
                    "department":      "All",
                    "section_heading": current_heading,
                    "page_number":     page_num,
                    "source_filename": filename,
                    "ingested_at":     ingested_at,
                },
            })

    return records


def load_docx(file_path: str) -> list[dict]:
    """
    Extract text from a DOCX file, grouping paragraphs under their
    nearest preceding Heading-style paragraph.
    """
    records = []
    filename = pathlib.Path(file_path).name
    doc_title = _infer_doc_title(filename)
    doc_type = _infer_doc_type(filename)
    ingested_at = datetime.now(timezone.utc).isoformat()

    doc = DocxDocument(file_path)
    current_heading = "General"
    current_section: list[str] = []
    section_index = 0

    def _flush():
        nonlocal section_index
        if current_section:
            records.append({
                "page_content": "\n".join(current_section),
                "metadata": {
                    "doc_title":       doc_title,
                    "doc_type":        doc_type,
                    "department":      "All",
                    "section_heading": current_heading,
                    "page_number":     section_index,
                    "source_filename": filename,
                    "ingested_at":     ingested_at,
                },
            })
            section_index += 1

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if para.style.name.startswith("Heading"):
            _flush()
            current_section = []
            current_heading = text
        else:
            current_section.append(text)

    _flush()  # flush final section
    return records


def load_txt(file_path: str) -> list[dict]:
    """
    Load a plain text or markdown file as a single record.
    Splits on double-newlines to create multiple records if the file is large.
    """
    filename = pathlib.Path(file_path).name
    doc_title = _infer_doc_title(filename)
    doc_type = _infer_doc_type(filename)
    ingested_at = datetime.now(timezone.utc).isoformat()

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split on double newlines for loose section grouping
    sections = [s.strip() for s in re.split(r"\n{2,}", full_text) if s.strip()]
    if not sections:
        return []

    records = []
    for idx, section in enumerate(sections, start=1):
        # Extract a heading from the first line if it looks like one
        first_line = section.split("\n")[0].strip()
        heading = first_line if _is_heading_line(first_line) or first_line.startswith("#") else "General"
        heading = heading.lstrip("#").strip()

        records.append({
            "page_content": section,
            "metadata": {
                "doc_title":       doc_title,
                "doc_type":        doc_type,
                "department":      "All",
                "section_heading": heading,
                "page_number":     idx,
                "source_filename": filename,
                "ingested_at":     ingested_at,
            },
        })

    return records


# ── Public API ─────────────────────────────────────────────────────

def load_document(file_path: str) -> list[dict]:
    """
    Route a file to the correct parser based on its extension.

    Supported formats:
      .pdf   — text-based PDF (pdfplumber)
      .docx  — Word document (python-docx)
      .txt   — plain text
      .md    — markdown

    Returns a list of page/section records ready for chunking.
    Raises ValueError for unsupported file types.
    """
    ext = pathlib.Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext in (".txt", ".md"):
        return load_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            "Accepted formats: .pdf, .docx, .txt, .md"
        )
