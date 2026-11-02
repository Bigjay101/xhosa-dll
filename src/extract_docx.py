"""P1.3 Paragraph extraction from the historical files (Member A).

Supports layout-preserving .docx files and plain .txt exports (see
extract_paragraphs_txt and extract_any at the end of this module).

Reads word/document.xml directly (zipfile + lxml) instead of python-docx's
Document.paragraphs, which skips text inside tables and text boxes.

Key behaviours (tested in tests/test_extract.py, T2):
  1. Every mc:Fallback element is removed first, so text boxes stored twice
     (DrawingML + VML fallback) are not duplicated.
  2. Every w:p is visited in document order, including paragraphs inside tables
     (w:tbl) and text boxes (w:txbxContent). Text belongs only to its nearest
     enclosing paragraph, so nested paragraphs are never double-counted.
  3. w:tab -> space, w:br / w:cr -> newline.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
TAG_P, TAG_T, TAG_TAB = f"{{{W}}}p", f"{{{W}}}t", f"{{{W}}}tab"
TAG_BR, TAG_CR = f"{{{W}}}br", f"{{{W}}}cr"
TAG_TBL, TAG_TXBX = f"{{{W}}}tbl", f"{{{W}}}txbxContent"
TAG_FALLBACK = f"{{{MC}}}Fallback"


def read_docx_part(docx_path: str | Path, part: str) -> bytes | None:
    """Return the raw bytes of one part of the .docx zip (e.g. 'docProps/core.xml')."""
    with zipfile.ZipFile(docx_path) as zf:
        try:
            return zf.read(part)
        except KeyError:
            return None


def _nearest_paragraph(el: etree._Element) -> etree._Element | None:
    node = el.getparent()
    while node is not None and node.tag != TAG_P:
        node = node.getparent()
    return node


def _has_ancestor(el: etree._Element, tag: str) -> bool:
    node = el.getparent()
    while node is not None:
        if node.tag == tag:
            return True
        node = node.getparent()
    return False


def _paragraph_text(p: etree._Element) -> str:
    parts: list[str] = []
    for el in p.iter(TAG_T, TAG_TAB, TAG_BR, TAG_CR):
        if _nearest_paragraph(el) is not p:
            continue  # belongs to a nested paragraph (e.g. a text box inside this one)
        if el.tag == TAG_T:
            parts.append(el.text or "")
        elif el.tag == TAG_TAB:
            parts.append(" ")
        else:
            parts.append("\n")
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def extract_paragraphs_from_xml(xml_bytes: bytes) -> list[dict]:
    """Extract paragraphs from the bytes of word/document.xml (see module docstring)."""
    root = etree.fromstring(xml_bytes)
    for fallback in list(root.iter(TAG_FALLBACK)):
        fallback.getparent().remove(fallback)
    rows = []
    for idx, p in enumerate(root.iter(TAG_P)):
        rows.append({
            "para_idx": idx,
            "text": _paragraph_text(p),
            "in_table": _has_ancestor(p, TAG_TBL),
            "in_textbox": _has_ancestor(p, TAG_TXBX),
        })
    return rows


def extract_paragraphs(docx_path: str | Path) -> list[dict]:
    """Return every paragraph of a .docx in document order:
    {para_idx, text, in_table, in_textbox}. Empty paragraphs are kept (text == "")
    because they mark block boundaries."""
    xml = read_docx_part(docx_path, "word/document.xml")
    if xml is None:
        raise ValueError(f"{docx_path}: no word/document.xml (not a .docx?)")
    return extract_paragraphs_from_xml(xml)


PAGE_RE = re.compile(r"[_\-. ]p(?:age)?[_\-]?(\d{1,4})$", re.IGNORECASE)


def page_number(path: str | Path) -> int | None:
    """Page number from a per-page file name, e.g. 'ISIG_1870-10-01_p005' -> 5; else None."""
    m = PAGE_RE.search(Path(path).stem)
    return int(m.group(1)) if m else None


def edition_stem(path: str | Path) -> str:
    """File stem with any page suffix removed: 'ISIG_1870-10-01_p005' -> 'ISIG_1870-10-01'."""
    return PAGE_RE.sub("", Path(path).stem)


def edition_id_from_path(path: str | Path, source: str) -> str:
    """Stable edition id: '<source>-<sanitised file stem without page suffix>', so all
    pages of one edition share one id (the archive's TXT export is one file per page)."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", edition_stem(path)).strip("_")
    return f"{source}-{stem}"


def read_text_file(path: str | Path) -> str:
    """Read a .txt file, trying UTF-8 (with or without BOM) first, then Windows-1252."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_paragraphs_txt(path: str | Path) -> list[dict]:
    """Return every line of a .txt edition as a paragraph record, in order:
    {para_idx, text, in_table=False, in_textbox=False}. Empty lines are kept
    (text == "") because they mark block boundaries; P2 merges short layout lines
    into blocks exactly as for .docx."""
    text = read_text_file(path).replace("\r\n", "\n").replace("\r", "\n")
    return [{"para_idx": i, "text": re.sub(r"[ \t]+", " ", line).strip(),
             "in_table": False, "in_textbox": False}
            for i, line in enumerate(text.split("\n"))]


SUPPORTED_SUFFIXES = (".docx", ".txt")


def extract_any(path: str | Path) -> list[dict]:
    """Dispatch on the file suffix: .docx -> extract_paragraphs, .txt -> extract_paragraphs_txt."""
    suffix = Path(path).suffix.lower()
    if suffix == ".docx":
        return extract_paragraphs(path)
    if suffix == ".txt":
        return extract_paragraphs_txt(path)
    raise ValueError(f"unsupported file type: {path}")
