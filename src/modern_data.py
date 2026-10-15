"""P3.1 / P5.1-P5.2 Modern corpora loaders (Member A).

Downloads are cached once to data/interim/ (modern_raw.parquet, vuk_langid.parquet)
so later runs, B, C and smoke runs work offline.

CHECK (P5.1): Vuk'uzenzele field names. text_field: auto picks the string column
with the longest mean length; set it explicitly in config.yaml once known.
CHECK (P5.2): the value MzansiText uses for isiXhosa in its `lang` column. Run
`python scripts/p5_modern.py --peek-mzansi` and set modern.mzansi.lang_value.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

VUK_SPLITS = ("train", "eval", "test")


def _pick_text_column(df: pd.DataFrame, configured: str = "auto") -> str:
    if configured != "auto":
        return configured
    obj = [c for c in df.columns if df[c].dtype == object]
    if not obj:
        raise ValueError(f"no string columns in {list(df.columns)}")
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())


def load_vuk(hf_id: str, config_name: str, text_field: str = "auto") -> pd.DataFrame:
    """Return Vuk'uzenzele rows as DataFrame[doc_id, split, text].

    Tries the dataset's own loader, then falls back to its parquet files."""
    from datasets import load_dataset  # heavy import kept local

    try:
        ds = load_dataset(hf_id, config_name)
    except Exception:  # noqa: BLE001 - script-based loading can fail on new versions
        files = {s: f"hf://datasets/{hf_id}/{config_name}/{s}-*.parquet" for s in VUK_SPLITS}
        ds = load_dataset("parquet", data_files=files)
    frames = []
    for split_name in ds:
        df = ds[split_name].to_pandas()
        col = _pick_text_column(df, text_field)
        frames.append(pd.DataFrame({
            "doc_id": [f"vuk-{config_name}-{split_name}-{i}" for i in range(len(df))],
            "split": split_name, "text": df[col].astype(str)}))
    out = pd.concat(frames, ignore_index=True)
    return out[out["text"].str.strip().astype(bool)].reset_index(drop=True)


def peek_mzansi_langs(hf_id: str, n: int = 25_000, split: str = "validation") -> Counter:
    """Count the `lang` values in the first n streamed rows (resolves the P5.2 CHECK).

    The validation split has about 19,400 rows and may be ordered by language, so
    the default reads all of it."""
    from datasets import load_dataset

    stream = load_dataset(hf_id, split=split, streaming=True)
    counts: Counter = Counter()
    for i, row in enumerate(stream):
        if i >= n:
            break
        counts[row.get("lang")] += 1
    return counts


def load_mzansi(hf_id: str, split: str, lang_value: str, max_rows: int = 200_000) -> pd.DataFrame:
    """Stream one MzansiText split and keep isiXhosa rows: DataFrame[doc_id, split, text].

    Streams at most `max_rows` rows in total, so the ~6 GB corpus is never downloaded."""
    if lang_value in (None, "", "CHECK"):
        raise ValueError("Set modern.mzansi.lang_value in config.yaml first "
                         "(run scripts/p5_modern.py --peek-mzansi).")
    from datasets import load_dataset

    stream = load_dataset(hf_id, split=split, streaming=True)
    rows = []
    for i, row in enumerate(stream):
        if i >= max_rows:
            break
        if row.get("lang") == lang_value and str(row.get("text", "")).strip():
            rows.append({"doc_id": f"mzansi-{split}-{i}", "split": split, "text": row["text"]})
    return pd.DataFrame(rows, columns=["doc_id", "split", "text"])
