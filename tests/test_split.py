"""T5 (Member A): edition-level splits and evaluation samples."""
import pandas as pd

from src.split import _pattern, sample_eval, split_editions


def meta():
    rows = []
    for source in ("isigidimi", "imvo"):
        for year in range(1880, 1886):
            for i in range(10):
                rows.append({"edition_id": f"{source}-{year}-{i}", "source": source, "year": year})
    rows.append({"edition_id": "imvo-x", "source": "imvo", "year": None})
    return pd.DataFrame(rows)


def test_each_edition_one_split_and_fractions():
    s = split_editions(meta(), (0.6, 0.2, 0.2), seed=13)
    assert s["edition_id"].is_unique and s["split"].notna().all()
    shares = s["split"].value_counts(normalize=True)
    assert abs(shares["train"] - 0.6) < 0.05 and abs(shares["test"] - 0.2) < 0.05


def test_every_year_in_every_split():
    s = split_editions(meta(), seed=13)
    per_year = s.dropna(subset=["year"]).groupby(["source", "year"])["split"].nunique()
    assert (per_year == 3).all()                  # 10 editions per year -> all three splits


def test_split_reproducible():
    a = split_editions(meta(), seed=13).sort_values("edition_id").reset_index(drop=True)
    b = split_editions(meta(), seed=13).sort_values("edition_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_sample_eval_caps_and_ranks():
    chunks = pd.DataFrame({"source": ["imvo"] * 50, "edition_id": [f"e{i % 5}" for i in range(50)]})
    rank = sample_eval(chunks, per_source=20, max_per_group=3, group_col="edition_id", seed=13)
    picked = chunks[rank >= 0]
    assert len(picked) == 15                      # 5 editions x cap 3 < 20
    assert picked.groupby("edition_id").size().max() == 3
    assert sorted(rank[rank >= 0]) == list(range(15))


def test_pattern_every_window_of_five_has_all_splits():
    pat = _pattern((0.6, 0.2, 0.2))
    assert pat.count("train") == 12 and pat.count("dev") == 4 and pat.count("test") == 4
    for start in range(len(pat)):
        window = [pat[(start + i) % len(pat)] for i in range(5)]
        assert set(window) == {"train", "dev", "test"}
