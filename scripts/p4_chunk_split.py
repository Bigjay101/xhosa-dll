"""P4 Chunks, edition-level splits, evaluation samples and the data freeze (Member A).

Outputs:
  data/processed/splits.csv            edition_id, source, year, split
  data/processed/hist_chunks.parquet   handbook D6.4 schema
  results/split_coverage.csv           chunks/words per source x year x split
  data/processed/MANIFEST.txt          sha256 of every processed file  -> commit this (M1)
Sanity check S4: no edition or identical chunk text in two splits; split word
shares within +/-3 points of 60/20/20.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from src.io_utils import append_sanity_row, get_logger, log_stage_stats, read_jsonl, write_parquet  # noqa: E402
from src.segment import make_chunks  # noqa: E402
from src.split import sample_eval, split_editions  # noqa: E402

COLUMNS = ["chunk_id", "cluster_id", "source", "edition_id", "date", "year", "split", "eval_set",
           "sample_rank", "block_id", "text", "n_words", "p_eng", "glotlid_agree", "eng_token_rate"]


def write_manifest(cfg: dict) -> Path:
    folder = P(cfg, "processed")
    lines = []
    for f in sorted(folder.glob("*")):
        if f.is_file() and f.name != "MANIFEST.txt":
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
    out = folder / "MANIFEST.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p4_chunk_split")
    blocks = [b for b in read_jsonl(P(cfg, "interim", "blocks_lang.jsonl")) if b["lang"] == "xho"]
    chunks = pd.DataFrame(make_chunks(blocks, cfg["segment"]["chunk_words"],
                                      carry=("date", "year", "p_eng", "glotlid_agree", "eng_token_rate")))
    log.info("%d chunks from %d isiXhosa blocks", len(chunks), len(blocks))

    fr = cfg["split"]["fractions"]
    splits = split_editions(chunks[["edition_id", "source", "year"]],
                            (fr["train"], fr["dev"], fr["test"]), seed=cfg["split"]["seed"])
    splits.to_csv(P(cfg, "processed", "splits.csv", mkdir=True), index=False)
    chunks = chunks.merge(splits[["edition_id", "split"]], on="edition_id", how="left")
    chunks["cluster_id"] = chunks["edition_id"]

    # identical chunk text in two splits: keep it in train only (S4)
    dup_text = chunks.groupby("text")["split"].nunique()
    cross = set(dup_text[dup_text > 1].index)
    if cross:
        before = len(chunks)
        chunks = chunks[~(chunks["text"].isin(cross) & (chunks["split"] != "train"))]
        log.warning("dropped %d dev/test chunks whose text also occurs in another split",
                    before - len(chunks))

    # P4.3 evaluation samples with a seeded sample_rank
    s = cfg["sample"]
    chunks["eval_set"], chunks["sample_rank"] = "none", -1
    for role, n in (("dev", s["dev_per_source"]), ("test", s["test_per_source"])):
        part = chunks[chunks["split"] == role]
        rank = sample_eval(part, n, s["max_per_edition"], "edition_id", seed=cfg["seed"])
        chunks.loc[rank.index[rank >= 0], "eval_set"] = role
        chunks.loc[rank.index, "sample_rank"] = rank
        got = (rank >= 0).groupby(part["source"]).sum().to_dict()
        log.info("%s sample per source: %s (target %d)", role, got, n)
        for src_name, k in got.items():
            if k < n:
                log.warning("%s: only %d %s chunks under the per-edition cap -- raise the cap "
                            "and log it if this matters (P4.3)", src_name, k, role)
    chunks = chunks[COLUMNS]
    write_parquet(chunks, P(cfg, "processed", "hist_chunks.parquet"))

    # S4 integrity
    ed_splits = chunks.groupby("edition_id")["split"].nunique().max()
    cross_after = (chunks.groupby("text")["split"].nunique() > 1).sum()
    shares = chunks.groupby("split")["n_words"].sum() / chunks["n_words"].sum()
    frac_ok = all(abs(shares.get(k, 0) - v) <= 0.03 for k, v in fr.items())
    status = "PASS" if (ed_splits == 1 and cross_after == 0 and frac_ok) else "WARN"
    if args.smoke and status == "WARN" and ed_splits == 1 and cross_after == 0:
        status = "SKIP"  # tiny smoke data cannot hit the fractions
    detail = (f"max splits per edition {ed_splits}; texts in >1 split {cross_after}; "
              f"word shares {shares.round(3).to_dict()}")
    append_sanity_row(cfg, "S4", status, detail)
    log.info("S4 %s %s", status, detail)

    cov = (chunks.groupby(["source", "year", "split"], dropna=False)
           .agg(n_chunks=("chunk_id", "size"), n_words=("n_words", "sum")).reset_index())
    cov.to_csv(P(cfg, "results", "split_coverage.csv", mkdir=True), index=False)
    log_stage_stats(cfg, "P4 chunks", chunks)
    log.info("manifest -> %s  (commit it; copy data/processed to the shared Drive; announce M1)",
             write_manifest(cfg))


if __name__ == "__main__":
    main()
