"""P3.6 Diagnose the language filter (Member A). Read-only; prints counts only, never text.

Question: are the blocks flagged as English really English, or is the filter (trained on
MODERN isiXhosa) flagging historical isiXhosa whose spelling looks unfamiliar? Removing
those would delete exactly the text this study is about.

Signals, all automatic:
  eng_token_rate  share of a block's words on the English word list (top-2000 English
                  words from Vuk'uzenzele eng, minus words in the isiXhosa lexicon).
                  Running English text scores roughly 0.4-0.7; isiXhosa scores near 0.
  p_eng           the filter's probability that the block is English.
  n_words         blocks under 32 words never become chunks, so only blocks of 32+
                  words matter for the experiments.
Writes results/p3_diagnose.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from src.io_utils import get_logger, read_jsonl  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--rate", type=float, default=0.2,
                        help="eng_token_rate a flagged block needs to count as English in the "
                             "'both signals' rule (default 0.2)")
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p3_diagnose")
    pd.set_option("display.width", 160)
    df = pd.DataFrame(read_jsonl(P(cfg, "interim", "blocks_lang.jsonl")))
    df["flag"] = df["p_eng"] >= cfg["langid"]["p_eng_threshold"]  # classifier alone
    df["chunkable"] = df["n_words"] >= cfg["segment"]["chunk_words"]
    df["both"] = df["flag"] & (df["eng_token_rate"] >= args.rate)
    df["toks"] = df["text"].str.split()
    df["digit_share"] = df["toks"].map(lambda ws: np.mean([any(c.isdigit() for c in w) for w in ws]))
    df["cap_share"] = df["toks"].map(lambda ws: np.mean([w[:1].isupper() for w in ws]))

    def wshare(g, col):
        return round(float((g["n_words"] * g[col]).sum() / max(1, g["n_words"].sum())), 4)

    rows = []
    for (source, chunkable), g in df.groupby(["source", "chunkable"]):
        f = g[g["flag"]]
        rows.append({
            "source": source, "blocks": "32+ words" if chunkable else "<32 words",
            "n_blocks": len(g), "words": int(g["n_words"].sum()),
            "flagged_blocks": len(f), "flagged_word_share": wshare(g, "flag"),
            "flagged_median_eng_token_rate": round(float(f["eng_token_rate"].median()), 3) if len(f) else None,
            "flagged_share_rate_under_0.05": round(float((f["eng_token_rate"] < 0.05).mean()), 3) if len(f) else None,
            "flagged_share_rate_over_0.3": round(float((f["eng_token_rate"] >= 0.3).mean()), 3) if len(f) else None,
            "unflagged_median_eng_token_rate": round(float(g.loc[~g["flag"], "eng_token_rate"].median()), 3),
            "flagged_median_digit_share": round(float(f["digit_share"].median()), 3) if len(f) else None,
            "flagged_median_capitalised_share": round(float(f["cap_share"].median()), 3) if len(f) else None,
            "unflagged_median_capitalised_share": round(float(g.loc[~g["flag"], "cap_share"].median()), 3),
            f"both_signals_word_share(rate>={args.rate})": wshare(g, "both"),
        })
    out = pd.DataFrame(rows).set_index(["source", "blocks"])
    out.to_csv(P(cfg, "results", "p3_diagnose.csv", mkdir=True))
    log.info("flagged-as-English blocks:\n%s", out.T.to_string())

    f = df[df["flag"]]
    bands = pd.cut(f["eng_token_rate"], [-0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    log.info("flagged words by eng_token_rate band (blocks of 32+ words):\n%s",
             f[f["chunkable"]].groupby([bands[f["chunkable"]], "source"], observed=False)["n_words"]
             .sum().unstack(fill_value=0).to_string())
    pbands = pd.cut(f["p_eng"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    log.info("flagged blocks by p_eng band:\n%s",
             f.groupby([pbands, "source"], observed=False).size().unstack(fill_value=0).to_string())
    by_year = (df[df["chunkable"]].groupby(["source", "year"])
               .apply(lambda g: pd.Series({"flagged_word_share": wshare(g, "flag"),
                                           "both_signals_word_share": wshare(g, "both")}),
                      include_groups=False))
    log.info("by year (blocks of 32+ words):\n%s", by_year.to_string())
    if df["glotlid_label"].notna().any():
        for name, g in (("all flagged", f), ("flagged, 32+ words", f[f["chunkable"]])):
            log.info("GlotLID says English for %.3f of %s blocks (%d)", float((g["glotlid_label"] == "eng").mean()),
                     name, len(g))
    log.info("final English share by words (lang == eng): %s",
             {s_: wshare(g, "is_eng") for s_, g in df.assign(is_eng=df["lang"] == "eng").groupby("source")})


if __name__ == "__main__":
    main()
