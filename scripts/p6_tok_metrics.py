"""P6 Tokenizer metrics, experiment X1 (Member A; B reruns it on new conditions).

  python scripts/p6_tok_metrics.py --check-vocab
  python scripts/p6_tok_metrics.py --condition H-orig --condition M1-orig --condition M2-orig
  python scripts/p6_tok_metrics.py --condition H-norm            # B, after M2

Outputs (rows for the same tokenizer x condition are replaced on rerun):
  results/X1_tok_metrics_chunks.parquet   one row per chunk x tokenizer x condition
  results/X1_summary.csv                  mean + cluster-bootstrap CI per metric
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from src.data_api import load_condition, load_lexicon  # noqa: E402
from src.io_utils import append_results_row, get_logger, write_parquet  # noqa: E402
from src.tok_metrics import chunk_tok_metrics, summarise  # noqa: E402


def load_tokenizers(names: list[str], allow_missing: bool, log) -> dict:
    out = {}
    try:
        from transformers import AutoTokenizer
    except ImportError:
        if allow_missing:
            log.warning("transformers not installed: X1 skipped")
            return out
        raise
    for name in names:
        try:
            out[name] = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # noqa: BLE001
            if not allow_missing:
                raise
            log.warning("could not load %s (%s): skipped", name, exc)
    return out


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--subset", default="core", choices=["core", "depth"])
    parser.add_argument("--tokenizers", nargs="*", default=None)
    parser.add_argument("--check-vocab", action="store_true")
    parser.add_argument("--allow-missing", action="store_true",
                        help="skip tokenizers that cannot be loaded (smoke/offline runs)")
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p6_tok_metrics")
    names = args.tokenizers or cfg["tokenizers"]
    toks = load_tokenizers(names, args.allow_missing or args.smoke, log)
    if not toks:
        return

    if args.check_vocab:
        a, b = cfg["models"]["xlmr"], cfg["models"]["afro"]
        if a in toks and b in toks:
            same = toks[a].get_vocab() == toks[b].get_vocab()
            log.info("P6.2 CHECK: XLM-R and AfroXLMR vocabularies identical: %s  (log this)", same)
        return

    lexicon = load_lexicon(cfg)
    table_path = P(cfg, "results", "X1_tok_metrics_chunks.parquet", mkdir=True)
    table = pd.read_parquet(table_path) if table_path.exists() else pd.DataFrame()
    for cond in args.condition:
        df = load_condition(cond, subset=args.subset, cfg=cfg)
        for name, tok in toks.items():
            rows = [{"chunk_id": r.chunk_id, "cluster_id": r.cluster_id, "source": r.source,
                     "tokenizer": name, "condition": cond, **chunk_tok_metrics(tok, r.text, lexicon)}
                    for r in df.itertuples()]
            new = pd.DataFrame(rows)
            if len(table):
                table = table[~((table["tokenizer"] == name) & (table["condition"] == cond))]
            table = pd.concat([table, new], ignore_index=True)
            log.info("%s on %s: fertility %.3f, pcw %.3f, oov %.3f (%d chunks)", name, cond,
                     new["fertility"].mean(), new["pcw"].mean(), new["oov"].mean(), len(new))
    write_parquet(table, table_path)
    summary = summarise(table, seed=cfg["seed"])
    summary.to_csv(P(cfg, "results", "X1_summary.csv"), index=False)
    for r in summary[summary["condition"].isin(args.condition)].itertuples():
        append_results_row(cfg, member="A", step="P6.3", experiment="X1", model=r.tokenizer,
                           condition=r.condition, subset=args.subset, n_chunks=r.n_chunks,
                           metric=r.metric, value=r.mean, ci_lo=r.ci_lo, ci_hi=r.ci_hi)


if __name__ == "__main__":
    main()
