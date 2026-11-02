"""Create a compact, paper-ready forest plot of orthographic intervention effects.

All point estimates and cluster-bootstrap 95% CIs are taken from the project's
X4 and X5 summary results; effects are signed after-minus-before NLL.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/mnt/data')

# Store the observed signed NLL differences (nats per original word).
# For historical normalisation, the project's delta is orig-minus-normalised,
# so negate both the estimate and interval to obtain after-minus-before.
conditions = [
    'Normalise\nhistorical text',
    'Historicise\nmodern text (M1)',
    'Random noise\non modern text (M1)',
]
models = {
    'XLM-R': [
        (-0.307247, -0.329837, -0.284660),
        ( 0.444171,  0.366527,  0.521833),
        ( 0.642200,  0.547557,  0.742145),
    ],
    'AfroXLMR': [
        (-0.399726, -0.424732, -0.374498),
        ( 0.648565,  0.553781,  0.744419),
        ( 0.803953,  0.703337,  0.904441),
    ],
}

def make_figure():
    # Use a single shared axis to ensure signed differences are comparable.
    fig, ax = plt.subplots(figsize=(3.50, 2.60))
    y_centers = np.array([2.0, 1.0, 0.0])
    model_offset = {'XLM-R': 0.14, 'AfroXLMR': -0.14}
    model_marker = {'XLM-R': 'o', 'AfroXLMR': 's'}

    # Plot a 95% cluster-bootstrap interval for every model/condition.
    for model, entries in models.items():
        centers = np.array([row[0] for row in entries], dtype=float)
        low = np.array([row[1] for row in entries], dtype=float)
        high = np.array([row[2] for row in entries], dtype=float)
        ys = y_centers + model_offset[model]
        errors = np.vstack((centers - low, high - centers))
        ax.errorbar(centers, ys, xerr=errors, fmt=model_marker[model],
                    ms=4.2, capsize=2.4, capthick=0.95, lw=1.05,
                    label=model)

    # Reference zero distinguishes reduced prediction loss from increased loss.
    ax.axvline(0, color='0.25', lw=0.9)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(conditions, fontsize=8.2)
    ax.set_xlim(-0.54, 1.0)
    ax.set_ylim(-0.46, 2.43)
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.tick_params(axis='x', labelsize=7.8)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.16, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'$\Delta$ per-word NLL (after $-$ before; nats)', fontsize=8)
    ax.text(-0.49, 2.36, 'Lower loss', ha='left', va='top', fontsize=7, color='0.32')
    ax.text(0.97, 2.36, 'Higher loss', ha='right', va='top', fontsize=7, color='0.32')
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.98, 0.84),
              fontsize=7.5, handlelength=1.6, borderaxespad=0.35)
    fig.subplots_adjust(left=0.40, right=0.98, top=0.96, bottom=0.22)
    fig.savefig(OUT / 'fig1_intervention_effects.pdf', bbox_inches='tight', pad_inches=0.045)
    fig.savefig(OUT / 'fig1_intervention_effects.png', dpi=260, bbox_inches='tight', pad_inches=0.045)
    plt.close(fig)

if __name__ == '__main__':
    make_figure()
