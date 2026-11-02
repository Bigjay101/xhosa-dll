"""P7.4 Build the smoke dataset from the REAL data (Member A, after M1).

Copies 5 editions per newspaper and small slices of the cached modern data into
data/smoke/, so `bash run_all.sh --smoke` runs every stage offline in minutes.
(Before the real data exists, use scripts/p0_make_synthetic.py instead.)
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from src.extract_docx import edition_id_from_path  # noqa: E402
from src.io_utils import get_logger  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--editions", type=int, default=5)
    parser.add_argument("--docs", type=int, default=30)
    args = parser.parse_args()
    real = load_config(args.config, smoke=False)
    smoke = load_config(args.config, smoke=True)
    log = get_logger(real, "p7_make_smoke")
    root = P(smoke, "raw").parent
    if root.exists():
        shutil.rmtree(root)
    for source, sub in real["historical"]["sources"].items():
        files = sorted(f for f in P(real, "raw", sub).rglob("*")
                       if f.is_file() and f.suffix.lower() in (".docx", ".txt"))
        editions: dict[str, list[Path]] = {}
        for f in files:  # keep all pages of an edition together
            editions.setdefault(edition_id_from_path(f, source), []).append(f)
        ids = sorted(editions)
        step = max(1, len(ids) // args.editions)
        pick = ids[::step][: args.editions]
        dest = P(smoke, "raw", sub)
        dest.mkdir(parents=True, exist_ok=True)
        for ed in pick:
            for f in editions[ed]:
                shutil.copy2(f, dest / f.name)
        log.info("%s: copied %d editions", source, len(pick))
    raw = pd.read_parquet(P(real, "interim", "modern_raw.parquet"))
    lid = pd.read_parquet(P(real, "interim", "vuk_langid.parquet"))
    raw_small = raw.groupby(["source", "split"], group_keys=False).head(args.docs)
    lid_small = lid.groupby("label", group_keys=False).head(300)
    P(smoke, "interim").mkdir(parents=True, exist_ok=True)
    raw_small.to_parquet(P(smoke, "interim", "modern_raw.parquet"), index=False)
    lid_small.to_parquet(P(smoke, "interim", "vuk_langid.parquet"), index=False)
    log.info("smoke data in %s -- now run: bash run_all.sh --smoke", root)


if __name__ == "__main__":
    main()
