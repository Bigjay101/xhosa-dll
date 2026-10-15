"""P1.4 Edition metadata (Member A).

Finds each edition's publication date, trying (in order):
  1. the file name (ISO dates, YYYY_MM_DD, DD-MM-YYYY, month names, bare years);
  2. docProps/core.xml dcterms:created -- usually the scan/OCR date, so it is
     used only if its year falls inside the publication range;
  3. a regex over the first ~30 paragraphs for an English-format date
     ("January 4, 1888", "4 January 1888") or the most common 18[789]x year.
CHECK (P1.4): confirm on 10 real files that the parsed years are plausible, and
extend FILENAME_PATTERNS for the naming pattern found in P1.2.
"""
from __future__ import annotations

import re
from collections import Counter

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
YEAR_RE = r"(18[5-9]\d|19[01]\d)"


def _month(name: str) -> int | None:
    s = name.lower().rstrip(".")
    return MONTHS.get(s, MONTHS.get(s[:3]))


def _valid(year: int, month: int | None = None, day: int | None = None,
           year_range: tuple[int, int] = (1870, 1894)) -> bool:
    if not (year_range[0] <= year <= year_range[1]):
        return False
    if month is not None and not 1 <= month <= 12:
        return False
    if day is not None and not 1 <= day <= 31:
        return False
    return True


def _fmt(year: int, month: int | None, day: int | None) -> str:
    if month and day:
        return f"{year:04d}-{month:02d}-{day:02d}"
    if month:
        return f"{year:04d}-{month:02d}"
    return f"{year:04d}"


def _from_text(text: str, year_range: tuple[int, int]) -> tuple[int, int | None, int | None] | None:
    """Find a full or partial date in free text (file names or masthead lines)."""
    t = text.lower()
    patterns = [
        (rf"{YEAR_RE}[-_. ](\d{{1,2}})[-_. ](\d{{1,2}})", ("y", "m", "d")),   # 1885-01-04
        (rf"(\d{{1,2}})[-_. ](\d{{1,2}})[-_. ]{YEAR_RE}", ("d", "m", "y")),   # 04-01-1885
        (rf"{MONTH_RE}\.?[ _-]*(\d{{1,2}})(?:st|nd|rd|th)?,?[ _-]*{YEAR_RE}", ("M", "d", "y")),
        (rf"(\d{{1,2}})(?:st|nd|rd|th)?[ _-]*{MONTH_RE}\.?,?[ _-]*{YEAR_RE}", ("d", "M", "y")),
        (rf"{MONTH_RE}\.?[ _-]*{YEAR_RE}", ("M", "y")),
        (rf"{YEAR_RE}[ _-]*{MONTH_RE}", ("y", "M")),
        (rf"{YEAR_RE}[-_. ](\d{{1,2}})(?!\d)", ("y", "m")),
    ]
    for pat, order in patterns:
        for m in re.finditer(pat, t):
            vals = dict(zip(order, m.groups()))
            year = int(vals["y"])
            if "M" in vals:
                month = _month(vals["M"])
            else:
                month = int(vals["m"]) if "m" in vals else None
            day = int(vals["d"]) if "d" in vals else None
            if _valid(year, month, day, year_range):
                return year, month, day
    return None


def parse_edition_meta(filename: str, core_xml: str | bytes | None,
                       first_paragraphs: list[str],
                       year_range: tuple[int, int] = (1870, 1894)) -> dict:
    """Return {date (YYYY-MM-DD | YYYY-MM | YYYY | None), year, month, meta_source}."""
    found = _from_text(filename.replace(".docx", ""), year_range)
    source = "filename"
    if found is None:
        m = re.search(rf"(?<!\d){YEAR_RE}(?!\d)", filename)
        if m and _valid(int(m.group(1)), year_range=year_range):
            found, source = (int(m.group(1)), None, None), "filename_year"
    if found is None and core_xml:
        xml = core_xml.decode("utf-8", "ignore") if isinstance(core_xml, bytes) else core_xml
        m = re.search(r"<dcterms:created[^>]*>(\d{4})-(\d{2})-(\d{2})", xml)
        if m and _valid(int(m.group(1)), year_range=year_range):
            found, source = (int(m.group(1)), int(m.group(2)), int(m.group(3))), "core_xml"
    if found is None and first_paragraphs:
        head = "\n".join(first_paragraphs[:30])
        found = _from_text(head, year_range)
        source = "masthead"
        if found is None:
            years = [int(y) for y in re.findall(rf"(?<!\d){YEAR_RE}(?!\d)", head)
                     if _valid(int(y), year_range=year_range)]
            if years:
                found, source = (Counter(years).most_common(1)[0][0], None, None), "masthead_year"
    if found is None:
        return {"date": None, "year": None, "month": None, "meta_source": "unresolved"}
    year, month, day = found
    return {"date": _fmt(year, month, day), "year": year, "month": month, "meta_source": source}
