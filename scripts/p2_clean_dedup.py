"""P2 Clean paragraphs into blocks, then deduplicate (Member A).

Outputs:
  data/interim/blocks.jsonl        {block_id, source, edition_id, date, year, block_idx, text, n_words}
  data/interim/blocks_dedup.jsonl  same, duplicates removed (earliest copy kept)
  results/dedup_report.csv         removals per source/year/kind
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.clean import build_blocks, normalise_unicode, word_counts  # noqa: E402
from src.config import P, base_parser, load_config  # noqa: E402
from src.dedup import dedup_blocks  # noqa: E402
from src.io_utils import get_logger, log_stage_stats, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p2_clean_dedup")
    paras = read_jsonl(P(cfg, "interim", "paragraphs_raw.jsonl"))
    eds = pd.read_csv(P(cfg, "interim", "editions.csv")).set_index("edition_id")
    counts = word_counts(normalise_unicode(p["text"]) for p in paras if p["text"])
    by_ed = defaultdict(list)
    for p in paras:
        by_ed[(p["source"], p["edition_id"])].append(p)

    blocks = []
    for (source, ed), ps in sorted(by_ed.items()):
        ps.sort(key=lambda p: p["para_idx"])
        meta = eds.loc[ed] if ed in eds.index else {}
        for b in build_blocks(ps, counts, cfg["clean"]["block_merge_if_median_below"],
                              cfg["clean"]["min_block_words"]):
            blocks.append({"block_id": f"{ed}-b{b['block_idx']:04d}", "source": source,
                           "edition_id": ed,
                           "date": None if pd.isna(meta.get("date", None)) else meta.get("date"),
                           "year": None if pd.isna(meta.get("year", None)) else int(meta.get("year")),
                           **b})
    write_jsonl(blocks, P(cfg, "interim", "blocks.jsonl", mkdir=True))
    log_stage_stats(cfg, "P2 blocks", pd.DataFrame(blocks))
    log.info("%d blocks", len(blocks))

    d = cfg["dedup"]
    kept, report = dedup_blocks(blocks, d["minhash_threshold"], d["shingle"], d["num_perm"],
                                d["exact_only_below_words"])
    write_jsonl(kept, P(cfg, "interim", "blocks_dedup.jsonl"))
    report.to_csv(P(cfg, "results", "dedup_report.csv", mkdir=True), index=False)
    log_stage_stats(cfg, "P2 dedup", pd.DataFrame(kept))
    log.info("kept %d of %d blocks; removed:\n%s", len(kept), len(blocks),
             report.groupby(["source", "kind"])[["n_removed", "words_removed"]].sum().to_string()
             if len(report) else "none")


if __name__ == "__main__":
    main()
