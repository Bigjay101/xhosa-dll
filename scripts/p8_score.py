
"""P8.3: Score isiXhosa chunks with pretrained language models.

Outputs follow the handbook D6.4 scoring schema.
Supports resumable XLM-R and AfroXLMR scoring.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Step 1: Add the repository root to the import path.
# This allows `python scripts/p8_score.py` to import modules from src/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.config import P, config_hash, load_config, set_seed
from src.data_api import load_condition
from src.mlm_scoring import score_mlm
from src.segment import words


# These are the exact columns required by the project handbook.
SCORE_COLUMNS = [
    "chunk_id", "cluster_id", "source", "model",
    "scorer", "condition", "total_nll", "n_tokens",
    "n_words_orig", "nll_per_word", "word_nll",
    "word_pieces", "runtime_s",
]


def parse_args():
    """Read the experiment settings supplied in the terminal."""

    # Step 1: Create an argument parser for this experiment.
    parser = argparse.ArgumentParser(
        description="Score historical or modern isiXhosa chunks"
    )

    # Step 2: Identify the model and dataset to evaluate.
    parser.add_argument("--model", choices=["xlmr", "afro", "mzansi"], required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--subset", choices=["core", "depth"], default="core")

    # Step 3: Support small debugging runs before full evaluation.
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)

    # Step 4: Support the shared configuration and smoke-data paths.
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--smoke", action="store_true")

    return parser.parse_args()


def load_model(model_name, cfg, device):
    """Load a pretrained masked language model and its tokenizer."""

    # Step 1: Look up the model repository in config.yaml.
    model_id = cfg["models"][model_name]

    # Step 2: Download or retrieve the matching fast tokenizer.
    # A fast tokenizer provides word_ids() for subword-to-word alignment.
    print(f"Loading tokenizer: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True
    )

    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required")

    # Step 3: Load the pretrained model with its masked-LM head.
    # The first run may download a substantial model checkpoint.
    print(f"Loading model: {model_id}", flush=True)
    model = AutoModelForMaskedLM.from_pretrained(model_id)

    # Step 4: Use float32 and disable dropout for reproducible scoring.
    model = model.float().to(device)
    model.eval()

    return model, tokenizer


def save_progress(rows, output_path):
    """Safely save completed scores so future runs can resume."""

    # Step 1: Build a DataFrame with the required scoring columns.
    df = pd.DataFrame(rows, columns=SCORE_COLUMNS)

    # Step 2: Ensure no text chunk has been scored twice.
    if df["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk IDs in scoring results")

    # Step 3: Write to a temporary file first.
    # This protects the previous checkpoint if writing is interrupted.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.parquet")
    df.to_parquet(temporary_path, index=False)

    # Step 4: Replace the previous score file only after writing succeeds.
    os.replace(temporary_path, output_path)


def record_completion(cfg, args, scores):
    """Log a completed experiment in results/results_log.csv."""

    # Step 1: Find the Git commit for reproducibility.
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=False
    )
    git_commit = commit.stdout.strip() if commit.returncode == 0 else "unknown"

    # Step 2: Record a descriptive mean.
    # Clustered bootstrap confidence intervals are calculated in P11.
    mean_nll = float(scores["nll_per_word"].mean())
    run_id = (
        f"P8_{args.model}_{args.condition}_{args.subset}_"
        f"{config_hash(cfg)}"
    )

    entry = {
        "run_id": run_id,
        "date": datetime.now(timezone.utc).isoformat(),
        "member": "B/C",
        "step": "P8",
        "experiment": "X2",
        "model": args.model,
        "condition": args.condition,
        "subset": args.subset,
        "n_chunks": len(scores),
        "metric": "mean_nll_per_word",
        "value": mean_nll,
        "ci_lo": None,
        "ci_hi": None,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit,
        "notes": "Descriptive mean; clustered CI pending P11",
    }

    # Step 3: Preserve previously logged experiments.
    log_path = P(cfg, "results", "results_log.csv", mkdir=True)
    old = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    # Step 4: Replace an existing record of this exact run,
    # rather than adding duplicate summary rows.
    if "run_id" in old.columns:
        old = old[old["run_id"] != run_id]

    updated = pd.concat([old, pd.DataFrame([entry])], ignore_index=True)
    updated.to_csv(log_path, index=False)

    print(f"Completed experiment. Mean NLL/word: {mean_nll:.4f}")


def main():
    """Execute and checkpoint the requested language-model experiment."""

    # Step 1: Read settings and establish a reproducible random seed.
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    set_seed(cfg["seed"])

    if args.model == "mzansi":
        raise NotImplementedError(
            "MzansiLM needs score_clm(); implement P8.2 first."
        )

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on this machine")

    batch_size = args.batch_size or cfg["scoring"]["batch_size"]

    # Step 2: Load the requested condition through the project's Data API.
    # This preserves the exact evaluation sample selected by Jordan.
    dataset = load_condition(args.condition, args.subset, cfg)
    if dataset.empty:
        raise ValueError("No chunks found for this condition")

    if dataset["chunk_id"].duplicated().any():
        raise ValueError("Input condition contains duplicate chunk IDs")

    # Step 3: Define the canonical output file.
    output_path = P(
        cfg, "scores",
        f"{args.model}__{args.condition}__{args.subset}.parquet",
        mkdir=True,
    )

    # Step 4: Recover valid previously scored chunks, if available.
    if output_path.exists():
        previous = pd.read_parquet(output_path)

        if not set(SCORE_COLUMNS).issubset(previous.columns):
            raise ValueError("Existing scores have an incompatible schema")

        if previous["chunk_id"].duplicated().any():
            raise ValueError("Existing scores contain duplicate chunk IDs")

        for column, expected in (
            ("model", args.model),
            ("condition", args.condition),
            ("scorer", cfg["scoring"]["method"]),
        ):
            if not previous[column].eq(expected).all():
                raise ValueError(
                    f"Existing scores use incompatible {column}"
                )

        if not set(previous["chunk_id"]).issubset(set(dataset["chunk_id"])):
            raise ValueError("Existing scores do not match the input dataset")

        completed = previous.to_dict("records")
    else:
        completed = []

    done_ids = {row["chunk_id"] for row in completed}
    pending = dataset[~dataset["chunk_id"].isin(done_ids)]

    # Step 5: Restrict new work for a quick pilot, if requested.
    if args.limit is not None:
        pending = pending.head(args.limit)

    print(
        f"Dataset: {len(dataset)} chunks; "
        f"already scored: {len(completed)}; "
        f"scoring now: {len(pending)}"
    )

    # Step 6: Avoid loading the large model if everything is finished.
    if not pending.empty:
        model, tokenizer = load_model(args.model, cfg, device)

        for index, row in enumerate(pending.itertuples(index=False), start=1):
            text = row.text

            # Validate that the input fits the configured context limit.
            encoded = tokenizer(
                words(text),
                is_split_into_words=True,
                truncation=False,
            )
            if len(encoded["input_ids"]) > cfg["scoring"]["max_tokens"]:
                raise ValueError(
                    f"Chunk {row.chunk_id} exceeds the token limit"
                )

            # Score one complete chunk; batch_size controls
            # the number of masked copies processed at once.
            start = time.perf_counter()
            result = score_mlm(
                model, tokenizer, [text],
                method=cfg["scoring"]["method"],
                batch_size=batch_size,
                device=device,
            )[0]
            runtime = time.perf_counter() - start

            # Keep the original word count as the denominator.
            # This makes transformed and original conditions comparable.
            original_words = int(row.n_words_orig)
            if original_words <= 0:
                raise ValueError("Cannot score a zero-word chunk")

            completed.append({
                "chunk_id": row.chunk_id,
                "cluster_id": row.cluster_id,
                "source": row.source,
                "model": args.model,
                "scorer": cfg["scoring"]["method"],
                "condition": args.condition,
                "total_nll": result["total_nll"],
                "n_tokens": result["n_tokens"],
                "n_words_orig": original_words,
                "nll_per_word": result["total_nll"] / original_words,
                "word_nll": result["word_nll"],
                "word_pieces": result["word_pieces"],
                "runtime_s": runtime,
            })

            # Save after a small number of chunks so an interruption
            # does not require scoring everything again.
            if index % args.save_every == 0:
                save_progress(completed, output_path)
                print(f"Saved {len(completed)} chunks", flush=True)

        # Step 7: Save the final partial or complete set of scores.
        save_progress(completed, output_path)

    # Step 8: Log summary statistics only for a fully completed subset.
    if len(completed) == len(dataset):
        record_completion(cfg, args, pd.DataFrame(completed))
    else:
        print(
            f"Partial result saved: {len(completed)}/{len(dataset)} chunks. "
            "Rerun without --limit to resume."
        )

    print(f"Score file: {output_path}")


if __name__ == "__main__":
    main()
