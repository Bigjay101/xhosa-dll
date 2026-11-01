"""P11.3-P11.4: word-level category, regression and context analyses (T15)."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from src.lexicon import lex_key


def assign_category(word: str, rewritten: bool, lexicon, english: set[str]) -> str:
    """First matching D1 category: NUMPUNCT, ENG, CAP, REWRITE, OOV, INLEX."""
    key = lex_key(word)
    if any(ch.isdigit() for ch in word) or not any(ch.isalpha() for ch in word):
        return "NUMPUNCT"
    if key in english:
        return "ENG"
    if word[0].isupper():
        return "CAP"
    if rewritten:
        return "REWRITE"
    if key not in lexicon:
        return "OOV"
    return "INLEX"


def category_contributions(hist_words: pd.DataFrame, mod_words: pd.DataFrame) -> pd.DataFrame:
    """Decompose the word-level gap so sum of contributions equals mean H minus M."""
    rows = []
    total_h, total_m = len(hist_words), len(mod_words)
    if not total_h or not total_m:
        raise ValueError("Both word tables must have observations")
    for category in ["NUMPUNCT", "ENG", "CAP", "REWRITE", "OOV", "INLEX"]:
        h = hist_words.loc[hist_words["category"] == category, "nll"]
        m = mod_words.loc[mod_words["category"] == category, "nll"]
        # The product of share and mean is simply category loss / all words.
        contribution = float(h.sum() / total_h - m.sum() / total_m)
        rows.append({"category": category, "p_h": len(h) / total_h, "p_m": len(m) / total_m,
                     "mean_h": float(h.mean()) if len(h) else np.nan,
                     "mean_m": float(m.mean()) if len(m) else np.nan,
                     "contribution": contribution})
    return pd.DataFrame(rows)


def explained_share(words: pd.DataFrame) -> dict:
    """Compare historical-indicator OLS coefficients before/after token controls.

    Standard errors and coefficient intervals are clustered by document/edition.
    """
    import statsmodels.api as sm

    required = {"nll", "hist", "pieces", "oov", "cap", "numpunct", "eng", "cluster_id"}
    if required - set(words.columns):
        raise ValueError(f"Missing word regression fields: {sorted(required-set(words.columns))}")
    if words["hist"].nunique() < 2:
        raise ValueError("Need historical and modern observations")
    data = words.copy()
    data["log2_pieces"] = np.log2(data["pieces"].clip(lower=1))
    # Prefix clusters by historical/modern status to avoid ID collisions.
    groups = data["hist"].astype(str) + ":" + data["cluster_id"].astype(str)
    raw_x = sm.add_constant(data[["hist"]].astype(float), has_constant="add")
    adj_x = sm.add_constant(data[["hist", "log2_pieces", "oov", "cap", "numpunct", "eng"]].astype(float), has_constant="add")
    raw = sm.OLS(data["nll"].astype(float), raw_x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    adjusted = sm.OLS(data["nll"].astype(float), adj_x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    b_raw, b_adj = float(raw.params["hist"]), float(adjusted.params["hist"])
    return {"b_raw": b_raw, "b_raw_ci_lo": float(raw.conf_int().loc["hist", 0]),
            "b_raw_ci_hi": float(raw.conf_int().loc["hist", 1]),
            "b_adj": b_adj, "b_adj_ci_lo": float(adjusted.conf_int().loc["hist", 0]),
            "b_adj_ci_hi": float(adjusted.conf_int().loc["hist", 1]),
            "explained": 1 - b_adj/b_raw if abs(b_raw) > 1e-12 else np.nan,
            "n_words": len(data)}


def context_gap(hist_words: pd.DataFrame, mod_words: pd.DataFrame,
                min_count: int = 5, n_boot: int = 10000, seed: int = 13) -> dict:
    """Compare shared identically spelled types; bootstrap whole word types."""
    def by_type(df):
        """Aggregate in-lex, uncapitalised forms with enough observations."""
        d = df.copy()
        if "inlex" in d:
            d = d[d["inlex"].astype(bool)]
        if "capitalized" in d:
            d = d[~d["capitalized"].astype(bool)]
        return d.groupby("word_key")["nll"].agg(["mean", "size"])

    h, m = by_type(hist_words), by_type(mod_words)
    shared = h.join(m, how="inner", lsuffix="_h", rsuffix="_m")
    shared = shared[(shared["size_h"] >= min_count) & (shared["size_m"] >= min_count)]
    deltas = (shared["mean_h"] - shared["mean_m"]).to_numpy(dtype=float)
    if not len(deltas):
        return {"n_types": 0, "estimate": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for start in range(0, n_boot, 500):
        end = min(n_boot, start + 500)
        ids = rng.integers(0, len(deltas), size=(end-start, len(deltas)))
        reps[start:end] = deltas[ids].mean(axis=1)
    lo, hi = np.quantile(reps, [.025, .975])
    return {"n_types": len(deltas), "estimate": float(deltas.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def dose_slope(share_changed, delta_nll) -> float:
    """Compute the least-squares line through the origin in D1."""
    s, d = np.asarray(share_changed, float), np.asarray(delta_nll, float)
    if s.shape != d.shape or not len(s) or np.dot(s, s) == 0:
        raise ValueError("Nonempty aligned rates and effects required")
    return float(np.dot(s, d) / np.dot(s, s))
