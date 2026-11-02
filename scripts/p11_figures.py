"""P11.6: render the paper's vector figures from complete experimental results."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import P, base_parser, load_config
from src.data_api import load_condition
from src.plots import fig1_main, fig2_spelling
from src.stats import cluster_bootstrap_mean


def main() -> None:
    """Compute condition means and create only figures supported by score files."""
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    means = []
    for model in ("xlmr", "afro"):
        for condition in ("H-orig", "M1-orig", "M2-orig", "H-norm",
                          "M1-hist", "M2-hist", "M1-noise", "M2-noise"):
            path = P(cfg, "scores", f"{model}__{condition}__core.parquet")
            if not path.exists():
                continue
            data = pd.read_parquet(path)
            if set(data.chunk_id) != set(load_condition(condition, cfg=cfg).chunk_id):
                print("Skipping incomplete figure score file:", path)
                continue
            mean, lo, hi = cluster_bootstrap_mean(
                data.nll_per_word, data.cluster_id,
                n_boot=cfg["stats"]["n_boot"], seed=cfg["seed"])
            means.append({"model": model, "condition": condition,
                          "estimate": mean, "ci_lo": lo, "ci_hi": hi, "n_chunks": len(data)})
    if means:
        summary = pd.DataFrame(means)
        summary.to_csv(P(cfg, "results", "X2_condition_means.csv", mkdir=True), index=False)
        out = P(cfg, "figures", "fig1_main.pdf", mkdir=True)
        fig1_main(summary, str(out))
        print("Saved", out)
    else:
        print("No completed score files: Figure 1 not generated")
    # Figure 2 is generated only when genuine per-family and dose tables exist.
    fpath = P(cfg, "results", "X4_families.csv")
    dpath = P(cfg, "results", "X5_dose.csv")
    family = pd.read_csv(fpath) if fpath.exists() else pd.DataFrame()
    dose = pd.read_csv(dpath) if dpath.exists() else pd.DataFrame()
    if not family.empty or not dose.empty:
        out = P(cfg, "figures", "fig2_spelling.pdf", mkdir=True)
        fig2_spelling(family, dose, str(out))
        print("Saved", out)
    else:
        print("Figure 2 deferred: real per-family/dose summaries not yet available")


if __name__ == "__main__":
    main()
