"""P1.5 Extraction checks S1 and S2 (Member A).

S1: whitespace-word totals per newspaper within +/-10% of the published figures.
S2: editions per year follow the publication history; totals near 202 and 518.
Writes results/editions_per_year.csv and logs PASS/WARN to results/sanity_log.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from src.io_utils import append_sanity_row, get_logger  # noqa: E402


def expected_editions_per_year() -> dict[str, dict[int, int]]:
    """Publication history from Schoots et al. (2023), handbook B4.1."""
    isig = {}
    for year in range(1870, 1889):
        months = range(10, 13) if year == 1870 else range(1, 13)
        n = 0
        for m in months:
            twice = (year == 1883 and m >= 7) or (year == 1884 and m <= 9)
            n += 2 if twice else 1
        isig[year] = 0 if year == 1886 else n
    imvo = {1884: 8, **{y: 52 for y in range(1885, 1895)}}
    return {"isigidimi": isig, "imvo": imvo}


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p1_check")
    eds = pd.read_csv(P(cfg, "interim", "editions.csv"))
    tag = " (smoke run: expected to differ)" if args.smoke else ""

    # S1 word totals
    for source, expected in cfg["historical"]["expected_words"].items():
        got = int(eds.loc[eds["source"] == source, "n_words"].sum())
        ratio = got / expected
        status = "PASS" if 0.9 <= ratio <= 1.1 else "WARN"
        if args.smoke:
            status = "SKIP"
        detail = f"{source}: {got:,} words vs published {expected:,} (ratio {ratio:.2f}){tag}"
        append_sanity_row(cfg, "S1", status, detail)
        log.info("S1 %s %s", status, detail)
        if status == "WARN":
            log.warning("ratio near 2 = duplicated text boxes; far below 1 = missed tables/text boxes")

    # S2 editions per year
    exp = expected_editions_per_year()
    rows, flags = [], []
    for source, by_year in exp.items():
        got = eds[eds["source"] == source].groupby("year").size()
        years = sorted(set(by_year) | {int(y) for y in got.index.dropna()})
        for y in years:
            e, o = by_year.get(y, 0), int(got.get(y, 0))
            flag = abs(o - e) > max(3, 0.25 * e)
            rows.append({"source": source, "year": y, "expected": e, "observed": o, "flag": flag})
            if flag:
                flags.append(f"{source} {y}: {o} vs ~{e}")
    pd.DataFrame(rows).to_csv(P(cfg, "results", "editions_per_year.csv"), index=False)
    totals = {s: int((eds["source"] == s).sum()) for s in exp}
    tot_ok = all(abs(totals[s] - cfg["historical"]["expected_editions"][s])
                 <= 0.1 * cfg["historical"]["expected_editions"][s] for s in exp)
    status = "PASS" if (tot_ok and not flags) else "WARN"
    if args.smoke:
        status = "SKIP"
    detail = f"totals {totals} (published 202/518); flagged years: {flags[:12]}{tag}"
    append_sanity_row(cfg, "S2", status, detail)
    log.info("S2 %s %s", status, detail)


if __name__ == "__main__":
    main()
