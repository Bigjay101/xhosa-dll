"""P1.3-P1.4 Extract paragraphs and edition metadata from the .docx or .txt files (Member A).

Files may be one per edition, or one per PAGE (e.g. ISIG_1870-10-01_p005.txt). Pages are
grouped into editions by file name and concatenated in page order, with an empty line
between pages, so every later step (and the edition-level split) sees whole editions.

Outputs:
  data/interim/paragraphs_raw.jsonl  {source, file, edition_id, para_idx, text, in_table, in_textbox}
  data/interim/editions.csv          one row per edition with date/year/meta_source
  results/meta_unresolved.csv        editions whose date could not be parsed (must be < 2%)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from collections import defaultdict  # noqa: E402

from src.extract_docx import (SUPPORTED_SUFFIXES, edition_id_from_path, edition_stem, extract_any,  # noqa: E402
                              page_number, read_docx_part)
from src.io_utils import get_logger, write_jsonl  # noqa: E402
from src.metadata import parse_edition_meta  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p1_extract")
    year_range = tuple(cfg["historical"]["year_range"])
    para_rows, ed_rows = [], []
    for source, sub in cfg["historical"]["sources"].items():
        files = sorted(f for f in P(cfg, "raw", sub).rglob("*")
                       if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
                       and not f.name.startswith("~$"))  # skip Word lock files
        kinds = sorted({f.suffix.lower() for f in files})
        log.info("%s: %d files %s", source, len(files), kinds)
        if len(kinds) > 1:
            log.warning("%s has both .docx and .txt files -- if they are the same editions "
                        "in two formats, keep only one format in data/raw/", source)
        by_edition: dict[str, list[Path]] = defaultdict(list)
        for f in files:
            by_edition[edition_id_from_path(f, source)].append(f)
        multi = sum(len(v) > 1 for v in by_edition.values())
        log.info("%s: %d editions (%d made of several page files)", source, len(by_edition), multi)
        for ed, pages in sorted(by_edition.items()):
            pages.sort(key=lambda f: (page_number(f) is None, page_number(f) or 0, f.name))
            paras, idx = [], 0
            for f in pages:
                try:
                    page_paras = extract_any(f)
                except Exception as exc:  # noqa: BLE001 - log and continue; investigate later
                    log.error("failed %s: %s", f, exc)
                    continue
                if paras:  # page boundary = block boundary
                    paras.append({"para_idx": idx, "text": "", "in_table": False, "in_textbox": False,
                                  "page": page_number(f), "file": f.name})
                    idx += 1
                for p in page_paras:
                    paras.append({**p, "para_idx": idx, "page": page_number(f), "file": f.name})
                    idx += 1
            if not paras:
                continue
            first = pages[0]
            core = read_docx_part(first, "docProps/core.xml") if first.suffix.lower() == ".docx" else None
            meta = parse_edition_meta(edition_stem(first) + first.suffix, core,
                                      [p["text"] for p in paras if p["text"]], year_range)
            for p in paras:
                para_rows.append({"source": source, "edition_id": ed, **p})
            page_nums = [page_number(f) for f in pages if page_number(f) is not None]
            ed_rows.append({"edition_id": ed, "source": source, "file": first.name, **meta,
                            "n_pages": len(pages),
                            "pages": ",".join(map(str, page_nums)),
                            "n_paragraphs": len(paras),
                            "n_words": sum(len(p["text"].split()) for p in paras)})
    n = write_jsonl(para_rows, P(cfg, "interim", "paragraphs_raw.jsonl", mkdir=True))
    eds = pd.DataFrame(ed_rows)
    eds.to_csv(P(cfg, "interim", "editions.csv"), index=False)
    unresolved = eds[eds["year"].isna()] if len(eds) else eds
    unresolved.to_csv(P(cfg, "results", "meta_unresolved.csv", mkdir=True), index=False)
    share = len(unresolved) / max(1, len(eds))
    if len(eds):
        log.info("pages per edition by source:\n%s",
                 eds.groupby("source")["n_pages"].describe()[["count", "mean", "min", "max"]].to_string())
    log.info("%d paragraphs from %d editions; unresolved dates: %d (%.1f%%)", n, len(eds),
             len(unresolved), 100 * share)
    if len(eds):
        log.info("date sources:\n%s", eds["meta_source"].value_counts().to_string())
    if share >= 0.02:
        log.warning("more than 2%% of editions have no date: extend src/metadata.py (P1.4)")


if __name__ == "__main__":
    main()
