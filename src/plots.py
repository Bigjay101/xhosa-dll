"""P11.6: compact, vector PDF figures with accessible markers and 95% CIs."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig1_main(summary: pd.DataFrame, out_path: str) -> None:
    """Plot mean NLL/word by model and condition, including clustered 95% CIs."""
    required = {"model", "condition", "estimate", "ci_lo", "ci_hi"}
    if required - set(summary.columns):
        raise ValueError("Figure 1 needs model/condition/estimate/CI fields")
    if summary.empty:
        raise ValueError("Cannot plot empty Figure 1")
    plot = summary.sort_values(["model", "condition"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, .32 * len(plot) + .8)))
    palette = {"xlmr": "#0072B2", "afro": "#D55E00", "mzansi": "#009E73"}
    for i, row in plot.iterrows():
        ax.errorbar(float(row.estimate), i,
                    xerr=[[max(0, row.estimate-row.ci_lo)], [max(0, row.ci_hi-row.estimate)]],
                    fmt="o", color=palette.get(row.model, "#333333"), markersize=5,
                    capsize=2, linewidth=1.3)
    ax.set_yticks(np.arange(len(plot)))
    ax.set_yticklabels([f"{row.model}: {row.condition}" for row in plot.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Per-word pseudo-NLL (nats), 95% cluster CI", fontsize=9)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig2_spelling(families: pd.DataFrame, dose: pd.DataFrame, out_path: str) -> None:
    """Two-panel spelling figure, plotting only measured effects and CIs."""
    if families.empty and dose.empty:
        raise ValueError("No spelling/dose results available for Figure 2")
    if not families.empty and not dose.empty:
        fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.6))
    else:
        fig, axis = plt.subplots(figsize=(4.2, 3.6))
        left, right = (axis, None) if not families.empty else (None, axis)
    if left is not None:
        selected = families.sort_values("estimate")
        lower = np.maximum(0, selected.estimate.to_numpy(float)-selected.ci_lo.to_numpy(float))
        upper = np.maximum(0, selected.ci_hi.to_numpy(float)-selected.estimate.to_numpy(float))
        left.errorbar(selected.estimate.to_numpy(float), np.arange(len(selected)),
                      xerr=np.vstack([lower, upper]), fmt="o", capsize=2,
                      color="#0072B2")
        left.axvline(0, color="black", linewidth=.7)
        left.set_yticks(np.arange(len(selected)))
        left.set_yticklabels(selected.family, fontsize=8)
        left.set_xlabel("Per-family gap closure (95% CI)", fontsize=9)
        left.set_title("A. Rule-family effects", fontsize=9)
    if right is not None:
        for condition, group in dose.groupby("condition"):
            group = group.sort_values("dose")
            right.errorbar(group.dose, group.estimate,
                           yerr=np.vstack([np.maximum(0, group.estimate-group.ci_lo),
                                           np.maximum(0, group.ci_hi-group.estimate)]),
                           fmt="o-", markersize=4, capsize=2,
                           label=condition)
        right.axhline(0, color="black", linewidth=.7)
        right.set_xlabel("Applied dose (eligible positions)", fontsize=9)
        right.set_ylabel("Increase in NLL/word", fontsize=9)
        right.set_title("B. Systematic vs random edits", fontsize=9)
        right.legend(fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
