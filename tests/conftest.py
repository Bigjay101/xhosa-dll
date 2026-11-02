"""Shared test fixtures. Makes `src` importable and provides a tiny synthetic
project (config + processed files) in a temporary directory."""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.lexicon import save_lexicon  # noqa: E402


@pytest.fixture
def tmp_cfg(tmp_path):
    """A config whose processed/conditions paths point into tmp_path, populated with
    small hist_chunks.parquet, modern_chunks.parquet, lexicon and English list."""
    cfg = load_config()
    for key in ("processed", "conditions", "results", "logs"):
        cfg["_paths"][key] = tmp_path / key
        (tmp_path / key).mkdir(parents=True, exist_ok=True)
    cfg["sample"]["depth_per_source"] = 3
    rows = []
    for source in ("isigidimi", "imvo"):
        for i in range(6):
            rows.append({"chunk_id": f"{source}:e{i % 3}:b{i}:0", "cluster_id": f"{source}-e{i % 3}",
                         "source": source, "edition_id": f"{source}-e{i % 3}", "date": "1885-01",
                         "year": 1885, "split": "test", "eval_set": "test" if i < 5 else "none",
                         "sample_rank": i if i < 5 else -1, "block_id": f"b{i}",
                         "text": " ".join(["teta"] * 32), "n_words": 32, "p_eng": 0.01,
                         "glotlid_agree": None, "eng_token_rate": 0.0})
    pd.DataFrame(rows).to_parquet(tmp_path / "processed" / "hist_chunks.parquet", index=False)
    mrows = [{"chunk_id": f"M1:d{i}:d{i}:0", "cluster_id": f"d{i}", "source": "M1", "doc_id": f"d{i}",
              "role": "test", "sample_rank": i, "text": " ".join(["thetha"] * 32), "n_words": 32}
             for i in range(5)]
    pd.DataFrame(mrows).to_parquet(tmp_path / "processed" / "modern_chunks.parquet", index=False)
    save_lexicon(Counter({"thetha": 10, "abantu": 5}), tmp_path / "processed" / "modern_lexicon.tsv")
    (tmp_path / "processed" / "english_wordlist.txt").write_text("the\nmeeting\n", encoding="utf-8")
    return cfg
