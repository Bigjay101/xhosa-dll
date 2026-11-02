"""T3 (Member A): deduplication keeps the earliest copy."""
from src.dedup import dedup_blocks

LONG = ("ivenkile entsha ithengisa iingubo imali encinane ukuthenga kwaye ukuthengisa iinkomo "
        "amanzi isikolo incwadi abantu bonke bayamenywa ngoku kakhulu")


def block(i, text, date):
    return {"block_id": f"b{i}", "source": "imvo", "edition_id": f"e{i}", "block_idx": 0,
            "date": date, "year": int(date[:4]), "text": text, "n_words": len(text.split())}


def test_exact_duplicate_keeps_earliest():
    kept, report = dedup_blocks([block(1, LONG, "1886-01-07"), block(2, LONG.upper() + "!", "1885-01-07")])
    assert [b["block_id"] for b in kept] == ["b2"]
    assert report["n_removed"].sum() == 1 and set(report["kind"]) == {"exact"}


def test_near_duplicate_removed_distinct_kept():
    near = LONG.replace("kakhulu", "namhlanje")
    other = "umntu uthetha khona phantsi phezulu " * 4
    kept, report = dedup_blocks([block(1, LONG, "1885-01-07"), block(2, near, "1885-01-14"),
                                 block(3, other, "1885-01-21")])
    assert {b["block_id"] for b in kept} == {"b1", "b3"}
    assert set(report["kind"]) == {"near"}
