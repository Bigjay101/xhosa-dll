"""P1.5b Diagnose S1/S2 warnings (Member A). Read-only: prints a report and writes
results/p1_diagnose_pages.csv (words per page file) and results/p1_diagnose_years.csv.

Checks:
  1. Editions with duplicate page numbers: two issues sharing one file-name date were merged.
  2. Editions with unusually many or few pages, compared with that newspaper's median.
  3. Words per page by newspaper and the emptiest page files: low S1 caused by blank or
     truncated pages shows up here.
  4. Words per page by year: low S1 concentrated in some years points to an export
     problem in those years; a uniform shortfall points to a counting or definition gap.
  5. A rough English share per newspaper (common English function words), since the
     published word counts may include English columns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from src.io_utils import get_logger  # noqa: E402

ENGLISH = {"the", "and", "of", "to", "in", "is", "that", "for", "was", "with", "be", "by",
           "on", "as", "at", "this", "which", "from", "have", "are", "not", "it", "his", "we"}


def _pages(cell) -> list[int]:
    return [int(p) for p in str(cell).split(",") if p.strip().isdigit()]


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p1_diagnose")
    pd.set_option("display.width", 160)
    eds = pd.read_csv(P(cfg, "interim", "editions.csv"))

    # 1. duplicate page numbers
    dups = [(r.edition_id, _pages(r.pages)) for r in eds.itertuples()
            if len(_pages(r.pages)) != len(set(_pages(r.pages)))]
    log.info("1. editions with duplicate page numbers (merged issues): %d", len(dups))
    for ed, pages in dups[:20]:
        log.info("   %s pages=%s", ed, pages)

    # 2. page-count outliers
    med = eds.groupby("source")["n_pages"].transform("median")
    odd = eds[(eds["n_pages"] > 1.5 * med) | (eds["n_pages"] < 0.5 * med)]
    log.info("2. page-count outliers (vs median per newspaper): %d\n%s", len(odd),
             odd[["edition_id", "date", "n_pages", "pages"]].to_string(index=False) if len(odd) else "")

    # 3. words per page file
    pr = pd.read_json(P(cfg, "interim", "paragraphs_raw.jsonl"), lines=True)
    pr["w"] = pr["text"].fillna("").str.split().str.len()
    per_file = pr.groupby(["source", "edition_id", "file"], as_index=False)["w"].sum()
    per_file.to_csv(P(cfg, "results", "p1_diagnose_pages.csv", mkdir=True), index=False)
    log.info("3. words per page file:\n%s",
             per_file.groupby("source")["w"].describe(percentiles=[.05, .5, .95]).round(0).to_string())
    for source, g in per_file.groupby("source"):
        thresh = 0.25 * g["w"].median()
        low = g[g["w"] < thresh].sort_values("w")
        log.info("   %s: %d page files under 25%% of the median (%.0f words); emptiest:\n%s",
                 source, len(low), thresh, low.head(10)[["file", "w"]].to_string(index=False))

    # 4. words per page by year
    eds["words_per_page"] = eds["n_words"] / eds["n_pages"].clip(lower=1)
    years = (eds.groupby(["source", "year"])
             .agg(editions=("edition_id", "size"), pages=("n_pages", "sum"),
                  words=("n_words", "sum"), words_per_page=("words_per_page", "mean"))
             .round(0).reset_index())
    years.to_csv(P(cfg, "results", "p1_diagnose_years.csv"), index=False)
    log.info("4. by year:\n%s", years.to_string(index=False))

    # 5. rough English share
    tok = pr[["source", "text"]].copy()
    tok["toks"] = tok["text"].fillna("").str.lower().str.findall(r"[a-z]+")
    tok = tok.explode("toks").dropna(subset=["toks"])
    share = tok.assign(eng=tok["toks"].isin(ENGLISH)).groupby("source")["eng"].mean()
    log.info("5. share of tokens that are common English function words "
             "(~0.25-0.30 for running English text):\n%s", share.round(3).to_string())

    expected = cfg["historical"]["expected_words"]
    got = eds.groupby("source")["n_words"].sum()
    for source, exp in expected.items():
        n_pages = int(eds.loc[eds["source"] == source, "n_pages"].sum())
        log.info("S1 gap %s: %s words missing, about %.0f words per page over %d pages",
                 source, f"{exp - int(got.get(source, 0)):,}",
                 (exp - got.get(source, 0)) / max(1, n_pages), n_pages)


if __name__ == "__main__":
    main()