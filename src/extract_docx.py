"""P1.3 Paragraph extraction from layout-preserving .docx files (Member A).

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


def edition_id_from_path(path: str | Path, source: str) -> str:
    """Stable edition id: '<source>-<sanitised file stem>'."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(path).stem).strip("_")
    return f"{source}-{stem}"
