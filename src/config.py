"""P0.3 Configuration helpers (Member A).

Every script loads the same config.yaml through `load_config`, which resolves all
paths relative to the repository root. With `smoke=True`, data and result paths
are redirected to data/smoke/<name> so smoke runs never overwrite real outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SMOKE_REDIRECT = ("raw", "interim", "processed", "conditions", "results", "scores", "figures", "logs")


def load_config(path: str | Path | None = None, smoke: bool = False) -> dict:
    """Load config.yaml and attach resolved paths under cfg['_paths'].

    Keys starting with '_' are runtime additions and are excluded from config_hash.
    """
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    cfg["_smoke"] = bool(smoke)
    smoke_dir = ROOT / cfg["paths"]["smoke"]
    paths = {}
    for key, rel in cfg["paths"].items():
        paths[key] = (smoke_dir / key) if (smoke and key in SMOKE_REDIRECT) else (ROOT / rel)
    if smoke:
        paths["scores"] = smoke_dir / "results" / "scores"
    cfg["_paths"] = paths
    return cfg


def P(cfg: dict, key: str, *parts: str, mkdir: bool = False) -> Path:
    """Return a resolved path: P(cfg, 'processed', 'hist_chunks.parquet').

    With mkdir=True the parent directory is created.
    """
    p = cfg["_paths"][key].joinpath(*parts)
    if mkdir:
        (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: WPS433 (optional dependency)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def config_hash(cfg: dict) -> str:
    """sha256 of the sorted YAML dump (runtime '_' keys excluded), first 10 chars."""
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    dump = yaml.safe_dump(clean, sort_keys=True)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()[:10]


def base_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the flags every script shares."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--smoke", action="store_true", help="run on data/smoke (handbook P7.4)")
    return parser


if __name__ == "__main__":
    c = load_config()
    print("config OK, hash", config_hash(c))
    for k, v in c["_paths"].items():
        print(f"  {k:<11} {v}")
