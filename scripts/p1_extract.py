"""P1.3-P1.4 Extract paragraphs and edition metadata from the .docx files (Member A).

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
from src.extract_docx import edition_id_from_path, extract_paragraphs, read_docx_part  # noqa: E402
from src.io_utils import get_logger, write_jsonl  # noqa: E402
from src.metadata import parse_edition_meta  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p1_extract")
    year_range = tuple(cfg["historical"]["year_range"])
    para_rows, ed_rows = [], []
    for source, sub in cfg["historical"]["sources"].items():
        files = sorted(P(cfg, "raw", sub).rglob("*.docx"))
        files = [f for f in files if not f.name.startswith("~$")]  # Word lock files
        log.info("%s: %d .docx files", source, len(files))
        for f in files:
            ed = edition_id_from_path(f, source)
            try:
                paras = extract_paragraphs(f)
            except Exception as exc:  # noqa: BLE001 - log and continue; investigate later
                log.error("failed %s: %s", f, exc)
                continue
            meta = parse_edition_meta(f.name, read_docx_part(f, "docProps/core.xml"),
                                      [p["text"] for p in paras if p["text"]], year_range)
            for p in paras:
                para_rows.append({"source": source, "file": f.name, "edition_id": ed, **p})
            ed_rows.append({"edition_id": ed, "source": source, "file": f.name, **meta,
                            "n_paragraphs": len(paras),
                            "n_words": sum(len(p["text"].split()) for p in paras)})
    n = write_jsonl(para_rows, P(cfg, "interim", "paragraphs_raw.jsonl", mkdir=True))
    eds = pd.DataFrame(ed_rows)
    eds.to_csv(P(cfg, "interim", "editions.csv"), index=False)
    unresolved = eds[eds["year"].isna()] if len(eds) else eds
    unresolved.to_csv(P(cfg, "results", "meta_unresolved.csv", mkdir=True), index=False)
    share = len(unresolved) / max(1, len(eds))
    log.info("%d paragraphs from %d editions; unresolved dates: %d (%.1f%%)", n, len(eds),
             len(unresolved), 100 * share)
    if len(eds):
        log.info("date sources:\n%s", eds["meta_source"].value_counts().to_string())
    if share >= 0.02:
        log.warning("more than 2%% of editions have no date: extend src/metadata.py (P1.4)")


if __name__ == "__main__":
    main()
