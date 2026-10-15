"""P1.2 Inventory of the raw historical files (Member A).

Writes results/data_inventory.csv (source, file, extension, size_kb).
CHECK: look at the naming pattern and whether dates appear in file names, and
record the answer in DATA_SOURCES.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from src.io_utils import get_logger  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p1_inventory")
    rows = []
    for source, sub in cfg["historical"]["sources"].items():
        folder = P(cfg, "raw", sub)
        if not folder.exists():
            log.warning("missing folder %s", folder)
            continue
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                rows.append({"source": source, "file": str(f.relative_to(folder)),
                             "extension": f.suffix.lower(), "size_kb": round(f.stat().st_size / 1024, 1)})
    df = pd.DataFrame(rows, columns=["source", "file", "extension", "size_kb"])
    out = P(cfg, "results", "data_inventory.csv", mkdir=True)
    df.to_csv(out, index=False)
    log.info("%d files -> %s", len(df), out)
    if len(df):
        log.info("by source/extension:\n%s", df.groupby(["source", "extension"]).size().to_string())
        log.info("first file names:\n%s", "\n".join(df["file"].head(10)))


if __name__ == "__main__":
    main()
