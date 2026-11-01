"""P11.2: compute the X2 historical gap, X4 normalisation and X5 historicisation.

Run AFTER P8 has completed each referenced score file. Never mix partial
checkpoints with complete datasets or silently mix model likelihood types.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.config import P, base_parser, load_config
from src.data_api import load_condition
from src.io_utils import append_results_row
from src.stats import cluster_bootstrap_diff
from src.stats_extra import (paired_cluster_bootstrap, sign_flip_test, bootstrap_p,
                             holm, ratio_ci)


def score_table(cfg: dict, model: str, condition: str, subset: str = "core") -> pd.DataFrame | None:
    """Return only complete, uniquely keyed score tables; missing files are skipped."""
    path = P(cfg, "scores", f"{model}__{condition}__{subset}.parquet")
    if not path.exists():
        print("MISSING:", path)
        return None
    data = pd.read_parquet(path)
    if data["chunk_id"].duplicated().any() or not np.isfinite(data["nll_per_word"]).all():
        raise ValueError(f"Invalid or duplicated scores: {path}")
    expected = load_condition(condition, subset=subset, cfg=cfg)
    if set(expected.chunk_id) != set(data.chunk_id):
        print(f"PARTIAL: {model}/{condition} has {len(data)} of {len(expected)} chunks; skipping")
        return None
    return data


def matched(base: pd.DataFrame, changed: pd.DataFrame) -> pd.DataFrame:
    """Align paired conditions by ID and assert no missing or duplicated pairs."""
    if set(base.chunk_id) != set(changed.chunk_id):
        raise ValueError("Paired conditions differ in chunk IDs")
    out = base[["chunk_id", "cluster_id", "source", "nll_per_word"]].merge(
        changed[["chunk_id", "nll_per_word"]], on="chunk_id", how="inner",
        validate="one_to_one", suffixes=("_orig", "_changed"))
    return out


def joint_gap_closure(base: pd.DataFrame, norm: pd.DataFrame, modern: pd.DataFrame,
                      n_boot: int, seed: int) -> tuple:
    """Resample H clusters ONCE for the numerator and denominator of GC.

    Independently resample modern-document clusters for the denominator. This
    preserves covariance between the paired normalisation gain and historical mean.
    """
    pair = matched(base, norm)
    h_clusters, h_inv = np.unique(pair.cluster_id.astype(str), return_inverse=True)
    m_clusters, m_inv = np.unique(modern.cluster_id.astype(str), return_inverse=True)
    h_sizes = np.bincount(h_inv)
    m_sizes = np.bincount(m_inv)
    orig_sums = np.bincount(h_inv, weights=pair.nll_per_word_orig)
    gains = pair.nll_per_word_orig - pair.nll_per_word_changed
    gain_sums = np.bincount(h_inv, weights=gains)
    modern_sums = np.bincount(m_inv, weights=modern.nll_per_word)
    rng = np.random.default_rng(seed)
    nums, dens = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        hi = rng.integers(0, len(h_clusters), size=len(h_clusters))
        mi = rng.integers(0, len(m_clusters), size=len(m_clusters))
        nums[i] = gain_sums[hi].sum() / h_sizes[hi].sum()
        dens[i] = orig_sums[hi].sum()/h_sizes[hi].sum() - modern_sums[mi].sum()/m_sizes[mi].sum()
    return nums, dens


def log_summary(cfg: dict, row: dict, experiment: str) -> None:
    """Record every estimated numeric effect in the standard results log."""
    append_results_row(cfg, run_id=f"{experiment}:{row['model']}:{row.get('reference', row.get('condition', ''))}:{row['metric']}",
                       member="C", step="P11.2", experiment=experiment,
                       model=row["model"], condition=row.get("condition", row.get("reference", "")),
                       subset="core", n_chunks=row.get("n_chunks", 0), metric=row["metric"],
                       value=row["estimate"], ci_lo=row["ci_lo"], ci_hi=row["ci_hi"],
                       notes="Cluster-bootstrap 95% CI; seed from config.yaml")


def main() -> None:
    """Write only summaries supported by complete score files; apply Holm to eight tests."""
    parser = base_parser(__doc__)
    parser.add_argument("--models", nargs="+", default=["xlmr", "afro"])
    parser.add_argument("--n-boot", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    n = args.n_boot or cfg["stats"]["n_boot"]
    seed = cfg["seed"]
    x2, x4, x5, primary = [], [], [], []
    for model in args.models:
        h = score_table(cfg, model, "H-orig")
        if h is None:
            continue
        modern = {ref: score_table(cfg, model, f"{ref}-orig") for ref in ("M1", "M2")}
        gaps = {}
        for ref, m in modern.items():
            if m is None:
                continue
            estimate, low, high, reps = cluster_bootstrap_diff(
                h.nll_per_word, h.cluster_id, m.nll_per_word, m.cluster_id,
                n_boot=n, seed=seed)
            row = {"model": model, "reference": ref, "condition": "H-orig", "metric": "G",
                   "n_chunks": len(h), "n_modern": len(m), "estimate": estimate,
                   "ci_lo": low, "ci_hi": high, "p_raw": bootstrap_p(reps)}
            x2.append(row)
            log_summary(cfg, row, "X2")
            gaps[ref] = (estimate, reps)
            if model in {"xlmr", "afro"}:
                primary.append((len(x2)-1, "X2", row["p_raw"]))
        hn = score_table(cfg, model, "H-norm")
        if hn is not None:
            aligned = matched(h, hn)
            delta = aligned.nll_per_word_orig - aligned.nll_per_word_changed
            est, lo, hi, reps = paired_cluster_bootstrap(delta, aligned.cluster_id, n_boot=n, seed=seed)
            row = {"model": model, "condition": "H-norm", "reference": "H-orig", "metric": "delta",
                   "n_chunks": len(aligned), "estimate": est, "ci_lo": lo, "ci_hi": hi,
                   "p_raw": sign_flip_test(delta, aligned.cluster_id, n_perm=cfg["stats"]["n_perm"], seed=seed)}
            x4.append(row)
            log_summary(cfg, row, "X4")
            if model in {"xlmr", "afro"}:
                primary.append((len(x4)-1, "X4", row["p_raw"]))
            for ref, m in modern.items():
                if m is None:
                    continue
                g = gaps[ref][0]
                if abs(g) < 1e-12:
                    print("Near-zero historical gap; GC is unstable:", model, ref)
                    continue
                num_reps, den_reps = joint_gap_closure(h, hn, m, n, seed)
                try:
                    rlo, rhi = ratio_ci(num_reps, den_reps)
                except ValueError:
                    print("Unstable GC denominator; reporting no GC CI:", model, ref)
                    rlo, rhi = np.nan, np.nan
                grow = {"model": model, "condition": "H-norm", "reference": ref, "metric": "GC",
                        "n_chunks": len(aligned), "estimate": est/g, "ci_lo": rlo, "ci_hi": rhi}
                x4.append(grow)
                log_summary(cfg, grow, "X4")
        for ref, m in modern.items():
            if m is None:
                continue
            for suffix, metric in (("hist", "eta"), ("noise", "nu")):
                transformed = score_table(cfg, model, f"{ref}-{suffix}")
                if transformed is None:
                    continue
                aligned = matched(m, transformed)
                effects = aligned.nll_per_word_changed - aligned.nll_per_word_orig
                est, lo, hi, _ = paired_cluster_bootstrap(effects, aligned.cluster_id, n_boot=n, seed=seed)
                row = {"model": model, "condition": f"{ref}-{suffix}", "reference": f"{ref}-orig",
                       "metric": metric, "n_chunks": len(aligned), "estimate": est,
                       "ci_lo": lo, "ci_hi": hi}
                if metric == "eta":
                    row["p_raw"] = sign_flip_test(effects, aligned.cluster_id,
                                                  n_perm=cfg["stats"]["n_perm"], seed=seed)
                x5.append(row)
                log_summary(cfg, row, "X5")
                if model in {"xlmr", "afro"} and ref == "M1" and metric == "eta":
                    primary.append((len(x5)-1, "X5", row["p_raw"]))
    # Pre-registered 8-test primary family; do not pretend the full Holm family
    # exists if any primary score table has not finished yet.
    if len(primary) == 8:
        adjusted = holm([item[2] for item in primary])
        for (index, experiment, _), p in zip(primary, adjusted):
            target = {"X2": x2, "X4": x4, "X5": x5}[experiment]
            target[index]["p_holm"] = p
    else:
        print(f"Primary Holm family incomplete ({len(primary)}/8); no adjusted p-values yet")
    for experiment, rows in (("X2", x2), ("X4", x4), ("X5", x5)):
        path = P(cfg, "results", f"{experiment}_summary.csv", mkdir=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(experiment, len(rows), "rows ->", path)


if __name__ == "__main__":
    main()
