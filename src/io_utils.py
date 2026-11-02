"""P0.4 Input/output helpers (Member A).

Small, dependable wrappers so every script reads and writes files the same way,
and so every reported number and sanity check lands in one log.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import P, config_hash

RESULTS_COLUMNS = ["run_id", "date", "member", "step", "experiment", "model", "condition",
                   "subset", "n_chunks", "metric", "value", "ci_lo", "ci_hi",
                   "config_hash", "git_commit", "notes"]
SANITY_COLUMNS = ["date", "check_id", "status", "detail"]
STATS_COLUMNS = ["stage", "source", "n_editions", "n_blocks", "n_chunks", "n_words", "notes"]


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def append_csv_row(path: str | Path, row: dict, columns: list[str]) -> None:
    """Append one row, writing the header if the file is new or empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in columns})


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=False, cwd=Path(__file__).resolve().parents[1])
        return out.stdout.strip() or "no-git"
    except OSError:
        return "no-git"


def append_results_row(cfg: dict, **row) -> None:
    """Record one reported number in results/results_log.csv (handbook D6.5)."""
    row.setdefault("date", dt.datetime.now().isoformat(timespec="seconds"))
    row.setdefault("config_hash", config_hash(cfg))
    row.setdefault("git_commit", git_commit())
    append_csv_row(P(cfg, "results", "results_log.csv"), row, RESULTS_COLUMNS)


def append_sanity_row(cfg: dict, check_id: str, status: str, detail: str) -> None:
    """Record a sanity check (S1-S8) as PASS or WARN in results/sanity_log.csv."""
    assert status in {"PASS", "WARN", "FAIL", "SKIP"}, status
    append_csv_row(P(cfg, "results", "sanity_log.csv"),
                   {"date": dt.datetime.now().isoformat(timespec="seconds"),
                    "check_id": check_id, "status": status, "detail": detail},
                   SANITY_COLUMNS)


def log_stage_stats(cfg: dict, stage: str, df: pd.DataFrame, notes: str = "") -> None:
    """Append per-source counts for one pipeline stage to results/X0_data_stats.csv."""
    for source, g in df.groupby("source"):
        row = {"stage": stage, "source": source,
               "n_editions": g["edition_id"].nunique() if "edition_id" in g else "",
               "n_blocks": g["block_id"].nunique() if "block_id" in g else "",
               "n_chunks": g["chunk_id"].nunique() if "chunk_id" in g else "",
               "n_words": int(g["n_words"].sum()) if "n_words" in g else "",
               "notes": notes}
        append_csv_row(P(cfg, "results", "X0_data_stats.csv"), row, STATS_COLUMNS)


def get_logger(cfg: dict, script_name: str) -> logging.Logger:
    """Logger writing to logs/<script>_<timestamp>.log and the console.

    The first lines record the config hash, git commit and smoke flag.
    """
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = P(cfg, "logs", f"{script_name}_{stamp}.log", mkdir=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.info("script=%s config_hash=%s git=%s smoke=%s", script_name, config_hash(cfg),
                git_commit(), cfg.get("_smoke"))
    return logger
