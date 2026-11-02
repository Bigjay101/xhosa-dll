"""P2.3 Diagnose deduplication (Member A). Read-only; prints counts only, never text.

Answers: are the removed duplicates the expected kind (repeated adverts, notices and
mastheads across editions) or a sign of a data problem (the same page exported twice)?

  1. Whole page files that are identical to another page file (a data problem if > 0).
  2. Size of the exact-duplicate blocks: share under 10 words (short boilerplate).
  3. How many copies each duplicated block has, and across how many editions.
  4. Share of duplicate copies that sit inside the SAME edition as another copy
     (long within-edition repeats point to repeated pages, not adverts).
  5. How long a repeated block keeps reappearing (first to last date): adverts typically
     run for weeks to months.
Writes results/p2_diagnose.csv with the per-source summary.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from src.dedup import exact_key  # noqa: E402
from src.io_utils import get_logger  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p2_diagnose")
    pd.set_option("display.width", 160)

    # 1. identical page files
    pr = pd.read_json(P(cfg, "interim", "paragraphs_raw.jsonl"), lines=True)
    pages = (pr.assign(text=pr["text"].fillna(""))
             .groupby(["source", "file"])["text"].apply(lambda s: " ".join(s.str.split().sum())))
    pages = pages[pages.str.split().str.len() >= 50]
    h = pages.map(lambda t: hashlib.md5(exact_key(t).encode("utf-8")).hexdigest())
    dup_pages = h[h.duplicated(keep=False)].rename("hash").reset_index()
    log.info("1. page files (>=50 words) identical to another page file: %d", len(dup_pages))
    for _, g in list(dup_pages.groupby("hash"))[:20]:
        log.info("   same text: %s", ", ".join(g["file"].tolist()[:6]))

    # 2-5. exact-duplicate blocks
    b = pd.read_json(P(cfg, "interim", "blocks.jsonl"), lines=True)
    b["key"] = b["text"].map(exact_key)
    b["copies"] = b.groupby("key")["key"].transform("size")
    b["date"] = pd.to_datetime(b["date"], errors="coerce")
    rows = []
    for source, g in b.groupby("source"):
        d = g[g["copies"] > 1]
        extra = d[d.duplicated("key")]  # the copies P2 removes (all but one per key)
        per_key = d.groupby("key").agg(copies=("key", "size"), editions=("edition_id", "nunique"),
                                       first=("date", "min"), last=("date", "max"),
                                       words=("n_words", "first"))
        same_ed = d.duplicated(["key", "edition_id"])
        span_days = (per_key["last"] - per_key["first"]).dt.days
        rows.append({
            "source": source,
            "blocks": len(g), "words": int(g["n_words"].sum()),
            "removed_exact_blocks": len(extra), "removed_exact_words": int(extra["n_words"].sum()),
            "removed_share_of_words": round(extra["n_words"].sum() / max(1, g["n_words"].sum()), 3),
            "removed_blocks_under_10w": round((extra["n_words"] < 10).mean(), 3),
            "removed_words_in_blocks_under_10w": round(
                extra.loc[extra["n_words"] < 10, "n_words"].sum() / max(1, extra["n_words"].sum()), 3),
            "median_removed_block_words": float(extra["n_words"].median()) if len(extra) else 0,
            "keys_2_copies": int((per_key["copies"] == 2).sum()),
            "keys_3_10_copies": int(per_key["copies"].between(3, 10).sum()),
            "keys_over_10_copies": int((per_key["copies"] > 10).sum()),
            "keys_over_100_copies": int((per_key["copies"] > 100).sum()),
            "share_copies_within_same_edition": round(same_ed.sum() / max(1, len(extra)), 3),
            "within_edition_copies_20w_plus": int((same_ed & (d["n_words"] >= 20)).sum()),
            "median_span_days": float(span_days.median()) if len(span_days) else 0,
            "share_keys_spanning_over_365_days": round((span_days > 365).mean(), 3) if len(span_days) else 0,
        })
    out = pd.DataFrame(rows).set_index("source")
    out.to_csv(P(cfg, "results", "p2_diagnose.csv", mkdir=True))
    log.info("2-5. exact-duplicate summary:\n%s", out.T.to_string())

    # removed words by block-length band
    b["band"] = pd.cut(b["n_words"], [0, 4, 9, 19, 49, 10**6], labels=["3-4", "5-9", "10-19", "20-49", "50+"])
    extra_all = b[(b["copies"] > 1) & b.duplicated("key")]
    log.info("removed exact-duplicate words by block length:\n%s",
             extra_all.pivot_table(index="band", columns="source", values="n_words",
                                   aggfunc="sum", observed=False).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()