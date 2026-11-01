"""P10.4: Create paired H-norm, M1/2-hist and M1/2-noise core conditions.

No scoring or H-test-based rule selection occurs here.  Every transformed row
retains the original chunk ID, cluster ID and original word denominator.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import pandas as pd
from src.config import P, load_config
from src.data_api import load_condition, load_lexicon, save_condition
from src.rule_runtime import load_frozen_rules
from src.normalise import normalise_text
from src.historicise import estimate_rates, historicise_text
from src.noise import add_letter_noise
from src.segment import words
from src.lexicon import oov_rate


def row_seed(seed: int, chunk_id: str) -> int:
    """Stable per-chunk random seed independent of Python hash randomisation."""
    return (seed + int.from_bytes(hashlib.sha256(chunk_id.encode()).digest()[:8], 'big')) % 2**32


def transform(cfg, source: str, dest: str, lexicon, rules, rates, kind: str,
              noise_rate: float | None = None, dose: float | None = None,
              families: list[str] | None = None) -> pd.DataFrame:
    """Transform complete core conditions, validate word alignment and save Parquet."""
    baseline = load_condition(source, cfg=cfg)
    records = []
    for row in baseline.to_dict('records'):
        text = row['text']
        seed = row_seed(cfg['seed'], row['chunk_id'])
        if kind == 'norm':
            updated, changes = normalise_text(text, lexicon, rules, families=families)
        elif kind == 'hist':
            updated, changes = historicise_text(text, rules, rates=rates if dose is None else None,
                                                dose=dose, families=families, seed=seed)
        elif kind == 'noise':
            updated, count = add_letter_noise(text, per_word_rate=noise_rate, seed=seed)
            changes = [{'pos': i, 'old': old, 'new': new, 'family': 'NOISE'}
                       for i, (old, new) in enumerate(zip(words(text), words(updated)))
                       if old != new]
            assert len(changes) == count
        else:
            raise ValueError('Unknown transformation kind')
        if len(words(updated)) != len(words(text)):
            raise ValueError(f'Word-count change in {dest}/{row["chunk_id"]}; do not score')
        records.append({**row, 'text': updated, 'condition': dest,
                        'n_words_new': len(words(updated)), 'n_changed': len(changes),
                        'rewrite_rate': len(changes) / int(row['n_words_orig']),
                        'changes_json': json.dumps(changes, ensure_ascii=False)})
    out = pd.DataFrame(records)
    save_condition(out, dest, cfg=cfg)
    print(f'{dest}: {len(out)} chunks, {out.n_changed.sum()} changed words, '
          f'rate {out.n_changed.sum()/out.n_words_orig.sum():.5f}')
    return out


def main() -> None:
    """Fit inverse rates on H-dev, then construct core experimental comparisons."""
    cfg = load_config()
    lexicon = load_lexicon(cfg)
    rules = load_frozen_rules(cfg['_root'])
    # Match dose to historical development spelling convention, never H-test.
    hdev = load_condition('H-dev', cfg=cfg)
    norm_logs, norm_text = [], []
    for text in hdev.text:
        updated, changes = normalise_text(text, lexicon, rules)
        norm_text.append(updated)
        norm_logs.append(changes)
    rates = estimate_rates(norm_logs, norm_text, rules)
    pd.DataFrame([{'family': key, 'matched_rate': value} for key, value in rates.items()])\
      .to_csv(P(cfg, 'results', 'family_rates.csv', mkdir=True), index=False)
    print('H-dev matched family rates:', rates)
    historical = transform(cfg, 'H-orig', 'H-norm', lexicon, rules, rates, 'norm')
    for ref in ['M1', 'M2']:
        modern = transform(cfg, f'{ref}-orig', f'{ref}-hist', lexicon, rules, rates, 'hist')
        observed = modern.n_changed.sum()/modern.n_words_orig.sum()
        noise = transform(cfg, f'{ref}-orig', f'{ref}-noise', lexicon, rules, rates,
                          'noise', noise_rate=float(observed))
        difference = abs(noise.n_changed.sum()/noise.n_words_orig.sum() - observed)
        print(f'{ref} matched noise check: abs rate difference {difference:.4f}; '
              f'{"PASS" if difference <= .01 else "WARN"} (target <= 0.01)')
    # Sanity S6: normalising modern words must change <2%.
    for ref in ['M1','M2']:
        source = load_condition(f'{ref}-orig', cfg=cfg)
        changes = sum(len(normalise_text(t, lexicon, rules)[1]) for t in source.text)
        print(f'S6 modern {ref}: {changes/source.n_words_orig.sum():.4f} (PASS <0.02)')
    print('S7 HISTORICAL OOV:', historical.text.map(lambda t:oov_rate(t, lexicon)).mean())
    print('P10 core conditions are written; score each condition with P8 before P11.')


if __name__ == '__main__':
    main()
