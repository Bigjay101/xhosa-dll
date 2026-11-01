"""P9.6-7: Tune validated orthographic rewrite candidates on H-dev and freeze.

Recomputes train-derived pattern support. No test-set texts or NLL scores are read.
The rewrite table is an exploratory family-level-validated dictionary, NOT a
human-verified mapping of each historical word's meaning.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter
from itertools import product
import hashlib
import pandas as pd
import yaml
from src.config import P, load_config
from src.data_api import load_condition, load_lexicon
from src.lexicon import is_oov, lex_key
from src.rules import count_patterns, edit_patterns, classify_pattern
from src.segment import words


def prepare_pool(candidates: pd.DataFrame, accepted: set[str], lexicon) -> pd.DataFrame:
    """Keep single-word in-lexicon targets supported by accepted P9.5 families."""
    patterns = count_patterns(candidates, min_support=1)
    support = {(r.op, r.src, r.tgt, r.left_char, r.right_char): int(r.support)
               for r in patterns.itertuples(index=False)}
    records = []
    for r in candidates.itertuples(index=False):
        if r.kind != 'edit' or ' ' in r.hist_form or r.hist_form in lexicon:
            continue
        if r.cand_form not in lexicon:
            continue
        edits = edit_patterns(r.hist_form, r.cand_form)
        families = {classify_pattern(*e) for e in edits}
        if len(families) != 1 or not edits:
            continue
        family = next(iter(families))
        if family not in accepted:
            continue
        records.append({'hist_form': r.hist_form, 'modern_form': r.cand_form,
                        'family': family, 'support': min(support[e] for e in edits),
                        'n_edits': len(edits), 'edit_distance': int(r.edit_distance),
                        'hist_freq': int(r.hist_freq), 'modern_freq': int(lexicon[r.cand_form])})
    return pd.DataFrame(records, columns=['hist_form','modern_form','family','support',
                                            'n_edits','edit_distance','hist_freq','modern_freq'])


def select_table(pool: pd.DataFrame, min_support: int, max_edits: int) -> pd.DataFrame:
    """Choose one deterministic highest-frequency candidate per historical word."""
    frame = pool[(pool.support >= min_support) & (pool.n_edits <= max_edits)].copy()
    return (frame.sort_values(['hist_form','modern_freq','edit_distance','modern_form'],
                              ascending=[True,False,True,True])
                 .drop_duplicates('hist_form').reset_index(drop=True))


def main() -> None:
    """Run the prespecified 12-configuration grid and save reproducible freeze files."""
    cfg = load_config()
    pairs = pd.read_parquet(P(cfg, 'processed', 'candidate_pairs.parquet'))
    validation = pd.read_csv(P(cfg, 'results', 'rule_validation.csv'))
    accepted = set(validation.loc[validation.accepted.fillna(False), 'family'])
    if not accepted:
        raise RuntimeError('P9.5 accepted no families; report RQ2 not measurable')
    lexicon = load_lexicon(cfg)
    dev = load_condition('H-dev', cfg=cfg)
    dev_words = [word for text in dev.text for word in words(text)]
    counts = Counter(lex_key(word) for word in dev_words if is_oov(word, lexicon))
    baseline = sum(counts.values()) / len(dev_words)
    pool = prepare_pool(pairs, accepted, lexicon)
    print('P9.5 accepted:', sorted(accepted), '| candidate pool:', len(pool))
    rows = []
    for minimum, edits, other in product([10,20,50], [1,2], [False,True]):
        # OTHER failed the validation criterion: grid branch remains disabled.
        selected = select_table(pool, minimum, edits)
        recovered = sum(counts.get(word, 0) for word in selected.hist_form)
        after = baseline - recovered / len(dev_words)
        rows.append({'min_support': minimum, 'max_edits_per_word': edits,
                     'include_other': other, 'n_rules': len(selected),
                     'dev_oov_before': baseline, 'dev_oov_after': after,
                     'dev_oov_reduction': baseline - after})
    tuning = pd.DataFrame(rows).sort_values(
        ['dev_oov_after','n_rules','min_support','max_edits_per_word','include_other'],
        ascending=[True,True,False,True,True]).reset_index(drop=True)
    tuning.to_csv(P(cfg, 'results', 'rule_tuning.csv', mkdir=True), index=False)
    chosen = tuning.iloc[0]
    final = select_table(pool, int(chosen.min_support), int(chosen.max_edits_per_word))
    delta_map = validation.set_index('family')['delta'].to_dict()
    final['validity_delta'] = final.family.map(delta_map)
    frozen = final[['hist_form','modern_form','family','support','validity_delta']]
    table_path = P(cfg, 'processed', 'rewrite_table_v1.tsv', mkdir=True)
    frozen.to_csv(table_path, sep='\t', index=False)
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    meta = {'version': 1, 'seed': cfg['seed'], 'accepted_families': sorted(accepted),
            'min_support': int(chosen.min_support),
            'max_edits_per_word': int(chosen.max_edits_per_word),
            'include_other': False, 'rewrite_table_sha256': digest,
            'method': 'train candidate dictionary; family-level context validation; H-dev OOV tuning',
            'limitations': 'Individual pair meanings unverified; no generalized regex rewrites; WD excluded'}
    target = Path(cfg['_root']) / 'configs/rules_v1.yaml'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(meta, sort_keys=False), encoding='utf-8')
    print('BEST P9.6:', chosen.to_dict())
    print('FROZEN MAPPINGS:', len(frozen), 'by family:', frozen.family.value_counts().to_dict())
    print('TABLE:', table_path, '| YAML:', target)
    print(frozen.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
