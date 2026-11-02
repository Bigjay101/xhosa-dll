"""P9.7: Read the frozen, training-derived historical-to-modern rewrite table.

Keep this module separate from src/rules.py, which contains the user's P9.3-5 code.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml


def load_frozen_rules(root: str | Path) -> dict:
    """Load approved families, rates, and lookup rows; reject inconsistent artifacts."""
    root = Path(root)
    with (root / 'configs/rules_v1.yaml').open(encoding='utf-8') as stream:
        meta = yaml.safe_load(stream)
    table = pd.read_csv(root / 'data/processed/rewrite_table_v1.tsv', sep='\t')
    required = {'hist_form', 'modern_form', 'family', 'support', 'validity_delta'}
    if not required.issubset(table):
        raise ValueError(f'Incomplete rewrite table: {required - set(table)}')
    accepted = set(meta['accepted_families'])
    if not set(table.family).issubset(accepted):
        raise ValueError('Rewrite table includes a family that failed validation')
    if table.hist_form.duplicated().any():
        raise ValueError('Ambiguous historical word mapping in frozen rewrite table')
    meta['table'] = table.to_dict('records')
    return meta
