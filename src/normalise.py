"""P10.1: conservative lexicon-guided historical spelling normalisation.

Uses the frozen train/dev-selected explicit mapping table, not unrestricted regex.
This narrower intervention protects against over-correction and is reported as
such in the paper.  Not a claim of individually verified translation.
"""
from __future__ import annotations

import re
from src.lexicon import lex_key
from src.segment import words


def _case_like(original: str, replacement: str) -> str:
    """Preserve initial capital or all-caps form where possible."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def normalise_text(text: str, lexicon, rules: dict, mode: str = 'lexicon',
                   families: list[str] | None = None) -> tuple[str, list[dict]]:
    """Replace known historical OOV keys with in-lexicon modern forms, logging edits.

    Only exact training-derived rewrite-table mappings are applied; `blind` mode
    is unsupported rather than inventing a non-equivalent generalised rewrite.
    Each edit preserves the number of whitespace words for paired P8 scoring.
    """
    if mode != 'lexicon':
        raise NotImplementedError('Blind regex ablation was not implemented; do not report it')
    permitted = set(families) if families is not None else set(rules['accepted_families'])
    lookup = {lex_key(row['hist_form']): row for row in rules['table']
              if row['family'] in permitted}
    output, changes = [], []
    for i, token in enumerate(words(text)):
        key = lex_key(token)
        row = lookup.get(key)
        if key in lexicon or not row or lex_key(row['modern_form']) not in lexicon:
            output.append(token)
            continue
        # Extract lexical core while leaving punctuation untouched.
        match = re.match(r'^([^\w]*)(.*?)([^\w]*)$', token, re.UNICODE)
        if not match:
            output.append(token)
            continue
        pre, core, post = match.groups()
        if not core:
            output.append(token)
            continue
        new = pre + _case_like(core, str(row['modern_form'])) + post
        output.append(new)
        if new != token:
            changes.append({'pos': i, 'old': token, 'new': new, 'family': row['family']})
    return ' '.join(output), changes
