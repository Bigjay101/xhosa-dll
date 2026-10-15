"""T7 (Member A): the data API returns standard columns and stable subsets."""
import pandas as pd
import pytest

from src import data_api
from src.data_api import STANDARD


def test_orig_conditions_standard_columns(tmp_cfg):
    h = data_api.load_condition("H-orig", cfg=tmp_cfg)
    m = data_api.load_condition("M1-orig", cfg=tmp_cfg)
    for df in (h, m):
        assert all(c in df.columns for c in STANDARD)
    assert len(h) == 10 and set(h["source"]) == {"isigidimi", "imvo"}   # eval_set == test only
    assert len(m) == 5


def test_depth_subset_stable_and_capped(tmp_cfg):
    a = data_api.load_condition("H-orig", subset="depth", cfg=tmp_cfg)
    b = data_api.load_condition("H-orig", subset="depth", cfg=tmp_cfg)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("source").size().max() == 3
    assert set(a["sample_rank"]) == {0, 1, 2}


def test_save_and_load_condition(tmp_cfg):
    base = data_api.load_condition("H-orig", cfg=tmp_cfg)
    with pytest.raises(ValueError):
        data_api.save_condition(base, "H-norm", cfg=tmp_cfg)       # missing B's columns
    built = base.assign(condition="H-norm", text=base["text"].str.replace("teta", "thetha"),
                        n_words_new=32, n_changed=32, rewrite_rate=1.0, changes_json="[]")
    data_api.save_condition(built, "H-norm", cfg=tmp_cfg)
    back = data_api.load_condition("H-norm", cfg=tmp_cfg)
    assert (back["text"].str.startswith("thetha")).all() and len(back) == len(base)
    with pytest.raises(ValueError):
        data_api.condition_path("bad code", cfg=tmp_cfg)


def test_lexicon_and_english(tmp_cfg):
    assert data_api.load_lexicon(cfg=tmp_cfg)["thetha"] == 10
    assert "meeting" in data_api.load_english_wordlist(cfg=tmp_cfg)
