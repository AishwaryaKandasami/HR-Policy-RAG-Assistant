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

def load_pdf(file_path: str) -> str:
    """
    Extract text from a text-based PDF and format it as Markdown.
    Uses font/positional heuristics (via pdfplumber) to identify headers.
    """
    filename = pathlib.Path(file_path).name
    md_lines = [f"# {filename}\n"]  # Default doc-level header
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            
            # Simple heuristic: find lines that look like headings
            # and prefix them with ##
            for line in text.split("\n"):
                stripped = line.strip()
                if _is_heading_line(stripped):
                    md_lines.append(f"\n## {stripped}\n")
                else:
                    md_lines.append(stripped)
            
            md_lines.append("\n") # Newline between pages
            
    return "\n".join(md_lines)


def load_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file and format as Markdown.
    Uses Word's native heading styles.
    """
    filename = pathlib.Path(file_path).name
    md_lines = [f"# {filename}\n"]
    
    doc = DocxDocument(file_path)
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        if para.style.name.startswith("Heading 1"):
            md_lines.append(f"\n## {text}\n")
        elif para.style.name.startswith("Heading"):
            md_lines.append(f"\n### {text}\n")
        else:
            md_lines.append(text)
            
    return "\n".join(md_lines)


def load_txt(file_path: str) -> str:
    """Load a plain text or markdown file as raw text."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ── Public API ─────────────────────────────────────────────────────

def load_document_to_markdown(file_path: str) -> str:
    """
    Route a file to the correct parser and return its structured Markdown.
    """
    ext = pathlib.Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext in (".txt", ".md"):
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'")
