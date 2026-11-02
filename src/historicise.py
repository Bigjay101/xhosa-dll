"""P10.2: introduce frozen historical spellings into modern text reproducibly."""
from __future__ import annotations

from collections import Counter
import random
import re
from src.lexicon import lex_key
from src.segment import words


def inverse_lookup(rules: dict, families=None) -> dict:
    """Create one deterministic historical candidate per modern form and family."""
    permitted = set(families) if families is not None else set(rules['accepted_families'])
    index = {}
    for row in sorted(rules['table'], key=lambda r: (-int(r['support']), r['hist_form'])):
        if row['family'] not in permitted:
            continue
        key = lex_key(str(row['modern_form']))
        index.setdefault(key, (str(row['hist_form']), row['family']))
    return index


def estimate_rates(h_dev_changes: list[list[dict]], h_dev_norm_texts: list[str],
                   rules: dict) -> dict[str, float]:
    """Fraction of inverse-eligible H-dev word positions with edits of each family."""
    if len(h_dev_changes) != len(h_dev_norm_texts):
        raise ValueError('H-dev logs and text rows must align')
    changed = Counter(log['family'] for logs in h_dev_changes for log in logs)
    inverse = inverse_lookup(rules)
    eligible = Counter()
    for text in h_dev_norm_texts:
        for word in words(text):
            if lex_key(word) in inverse:
                eligible[inverse[lex_key(word)][1]] += 1
    return {family: min(1.0, changed[family]/eligible[family]) if eligible[family] else 0.0
            for family in rules['accepted_families']}


def historicise_text(text: str, rules: dict, rates: dict[str,float] | None = None,
                      dose: float | None = None, families: list[str] | None = None,
                      seed: int = 13) -> tuple[str, list[dict]]:
    """Invert selected mappings with Bernoulli rate or fixed dose per eligible word."""
    if (rates is None) == (dose is None):
        raise ValueError('Provide exactly one of rates or dose')
    if dose is not None and not 0 <= dose <= 1:
        raise ValueError('dose must be between 0 and 1')
    index = inverse_lookup(rules, families)
    rng = random.Random(seed)
    output, changes = [], []
    for i, token in enumerate(words(text)):
        entry = index.get(lex_key(token))
        if not entry:
            output.append(token)
            continue
        historical, family = entry
        probability = dose if dose is not None else float(rates.get(family, 0.0))
        if not 0 <= probability <= 1:
            raise ValueError('Invalid family matched rate')
        if rng.random() >= probability:
            output.append(token)
            continue
        match = re.match(r'^([^\w]*)(.*?)([^\w]*)$', token, re.UNICODE)
        if not match:
            output.append(token)
            continue
        pre, core, post = match.groups()
        if core.isupper():
            historical = historical.upper()
        elif core[:1].isupper():
            historical = historical.capitalize()
        new = pre + historical + post
        output.append(new)
        if new != token:
            changes.append({'pos': i, 'old': token, 'new': new, 'family': family})
    return ' '.join(output), changes
