"""P7.2 Data API (Member A) -- the ONLY way analysis code reads data.

B and C load every text set through `load_condition`, so they never touch raw
files and every comparison uses exactly the same chunks.

Condition codes (handbook D6.4):
  built from the frozen chunks : H-orig, M1-orig, M2-orig, H-dev, M1-dev, M2-dev
  built by B (data/conditions/): H-norm, H-blind, H-norm.ASP, M1-hist, M1-hist.ASP,
                                 M1-hist.d050, M1-noise, M1-noise.d050, ...
Standard columns returned: chunk_id, cluster_id, source, text, n_words_orig, sample_rank.
subset='core' returns every row; subset='depth' returns the first
sample.depth_per_source rows per source by sample_rank (identical for everyone).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import P, load_config
from src.lexicon import load_lexicon_file

STANDARD = ["chunk_id", "cluster_id", "source", "text", "n_words_orig", "sample_rank"]
CODE_RE = re.compile(r"^(H|M1|M2)-[a-z]+(\.[A-Za-z0-9]+)?$")
ORIG = {"H-orig": ("hist", "test", None), "H-dev": ("hist", "dev", None),
        "M1-orig": ("modern", "test", "M1"), "M1-dev": ("modern", "dev", "M1"),
        "M2-orig": ("modern", "test", "M2"), "M2-dev": ("modern", "dev", "M2")}


def _cfg(cfg: dict | None) -> dict:
    return cfg if cfg is not None else load_config()


def load_hist_chunks(eval_set: str | None = None, source: str | None = None,
                     cfg: dict | None = None) -> pd.DataFrame:
    """All historical chunks (every split), optionally filtered by eval_set
    ('dev' | 'test' | 'none') and source ('isigidimi' | 'imvo')."""
    df = pd.read_parquet(P(_cfg(cfg), "processed", "hist_chunks.parquet"))
    if eval_set is not None:
        df = df[df["eval_set"] == eval_set]
    if source is not None:
        df = df[df["source"] == source]
    return df.reset_index(drop=True)


def load_modern_chunks(source: str | None = None, role: str | None = None,
                       cfg: dict | None = None) -> pd.DataFrame:
    """Modern chunks, optionally filtered by source ('M1' | 'M2') and role ('dev' | 'test')."""
    df = pd.read_parquet(P(_cfg(cfg), "processed", "modern_chunks.parquet"))
    if source is not None:
        df = df[df["source"] == source]
    if role is not None:
        df = df[df["role"] == role]
    return df.reset_index(drop=True)


def depth_subset(df: pd.DataFrame, n: int | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """First n rows per source by sample_rank (default: sample.depth_per_source)."""
    n = n if n is not None else _cfg(cfg)["sample"]["depth_per_source"]
    return (df.sort_values(["source", "sample_rank"]).groupby("source", sort=True).head(n)
            .reset_index(drop=True))


def condition_path(code: str, cfg: dict | None = None) -> Path:
    if not CODE_RE.match(code):
        raise ValueError(f"bad condition code {code!r} (see handbook D6.4)")
    return P(_cfg(cfg), "conditions", f"{code}.parquet")


def load_condition(code: str, subset: str = "core", cfg: dict | None = None) -> pd.DataFrame:
    """Load one condition with the standard columns (plus any extras B saved)."""
    cfg = _cfg(cfg)
    if subset not in {"core", "depth"}:
        raise ValueError("subset must be 'core' or 'depth'")
    if code in ORIG:
        kind, role, src = ORIG[code]
        if kind == "hist":
            df = load_hist_chunks(eval_set=role, cfg=cfg)
        else:
            df = load_modern_chunks(source=src, role=role, cfg=cfg)
        df = df.rename(columns={"n_words": "n_words_orig"})
        df = df[STANDARD].copy()
        df["condition"] = code
    else:
        path = condition_path(code, cfg)
        if not path.exists():
            raise FileNotFoundError(f"{path} not built yet (B builds it in P10.4)")
        df = pd.read_parquet(path)
    df = df.sort_values(["source", "sample_rank"]).reset_index(drop=True)
    return depth_subset(df, cfg=cfg) if subset == "depth" else df


def save_condition(df: pd.DataFrame, code: str, cfg: dict | None = None) -> Path:
    """Validate and save a condition built by B: standard columns + condition,
    n_words_new, n_changed, rewrite_rate, changes_json."""
    required = STANDARD + ["condition", "n_words_new", "n_changed", "rewrite_rate", "changes_json"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"condition {code}: missing columns {missing}")
    if df["chunk_id"].duplicated().any():
        raise ValueError(f"condition {code}: duplicate chunk_id")
    path = condition_path(code, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_lexicon(cfg: dict | None = None) -> Counter:
    return load_lexicon_file(P(_cfg(cfg), "processed", "modern_lexicon.tsv"))


def load_english_wordlist(cfg: dict | None = None) -> set[str]:
    path = P(_cfg(cfg), "processed", "english_wordlist.txt")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
