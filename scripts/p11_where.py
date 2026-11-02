"""P11.3 word-level gap decomposition, regression and shared-type context gaps."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.analysis import (assign_category, category_contributions,
                          context_gap, explained_share)
from src.config import P, base_parser, load_config
from src.data_api import load_condition, load_lexicon, load_english_wordlist
from src.lexicon import lex_key
from src.segment import words


def word_table(cfg: dict, model: str, condition: str, lexicon, english, rewritten_ids: dict | None = None) -> pd.DataFrame:
    """Join word-level NLLs to their exact input words without accessing raw data."""
    scores_path = P(cfg, "scores", f"{model}__{condition}__core.parquet")
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    scores = pd.read_parquet(scores_path)
    data = load_condition(condition, cfg=cfg)
    joined = data.merge(scores[["chunk_id", "word_nll", "word_pieces"]], on="chunk_id", validate="one_to_one")
    if len(joined) != len(data):
        raise ValueError("Full matching score coverage required")
    out = []
    for row in joined.itertuples():
        ws = words(row.text)
        if len(ws) != len(row.word_nll) or len(ws) != len(row.word_pieces):
            raise ValueError(f"Word alignment mismatch: {row.chunk_id}")
        changed = (rewritten_ids or {}).get(row.chunk_id, set())
        for i, (word, loss, pieces) in enumerate(zip(ws, row.word_nll, row.word_pieces)):
            category = assign_category(word, i in changed, lexicon, english)
            out.append({"chunk_id": row.chunk_id, "cluster_id": row.cluster_id,
                        "word_key": lex_key(word), "category": category, "nll": float(loss),
                        "pieces": int(pieces), "hist": int(condition.startswith("H-")),
                        "oov": int(lex_key(word) not in lexicon), "cap": int(word[:1].isupper()),
                        "numpunct": int(category == "NUMPUNCT"), "eng": int(category == "ENG"),
                        "inlex": lex_key(word) in lexicon, "capitalized": word[:1].isupper()})
    return pd.DataFrame(out)


def main() -> None:
    """Write reproducible X3 tables using scores and the standard data API."""
    parser = base_parser(__doc__)
    parser.add_argument("--models", nargs="+", default=["xlmr", "afro"])
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    lexicon, english = load_lexicon(cfg), load_english_wordlist(cfg)
    outputs = {"decomposition": [], "regression": [], "context_gap": []}
    for model in args.models:
        try:
            # Tag original historical word positions changed by H-norm, if
            # the condition exists; otherwise REWRITE is explicitly absent.
            import json
            norm_path = P(cfg, "conditions", "H-norm.parquet")
            rewrite_map = {}
            if norm_path.exists():
                change_table = pd.read_parquet(norm_path)
                rewrite_map = {row.chunk_id: {int(e["pos"]) for e in json.loads(row.changes_json)}
                               for row in change_table.itertuples(index=False)}
            h = word_table(cfg, model, "H-orig", lexicon, english,
                           rewritten_ids=rewrite_map)
        except (FileNotFoundError, ValueError) as exc:
            print("Cannot analyse", model, ":", exc)
            continue
        for ref in ("M1", "M2"):
            try:
                m = word_table(cfg, model, f"{ref}-orig", lexicon, english)
            except (FileNotFoundError, ValueError) as exc:
                print("Skipping", model, ref, exc)
                continue
            decomp = category_contributions(h, m)
            decomp["model"], decomp["reference"] = model, ref
            outputs["decomposition"].append(decomp)
            try:
                reg = explained_share(pd.concat([h, m], ignore_index=True))
                outputs["regression"].append({"model": model, "reference": ref, **reg})
            except Exception as exc:
                print("Regression unavailable", model, ref, exc)
            outputs["context_gap"].append({"model": model, "reference": ref,
                                           "condition": "H-orig", **context_gap(h, m, seed=cfg["seed"])})
            if norm_path.exists() and P(cfg, "scores", f"{model}__H-norm__core.parquet").exists():
                try:
                    hn = word_table(cfg, model, "H-norm", lexicon, english)
                    outputs["context_gap"].append({"model": model, "reference": ref,
                                                   "condition": "H-norm", **context_gap(hn, m, seed=cfg["seed"])})
                except ValueError as exc:
                    print("Skipping incomplete H-norm context:", exc)
    for name, records in outputs.items():
        path = P(cfg, "results", f"X3_{name}.csv", mkdir=True)
        (pd.concat(records, ignore_index=True) if name == "decomposition" and records
         else pd.DataFrame(records)).to_csv(path, index=False)
        print("Saved", path)


if __name__ == "__main__":
    main()
