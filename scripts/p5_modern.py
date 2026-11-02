"""P5 Modern corpora, lexicon and modern chunks (Member A). Also caches the
Vuk'uzenzele eng/xho texts used to train the language filter in P3.

  python scripts/p5_modern.py --peek-mzansi   # resolve the P5.2 CHECK (lang value)
  python scripts/p5_modern.py                 # download (first time), cache, build outputs
  python scripts/p5_modern.py --offline       # rebuild from the cache only

Outputs:
  data/interim/modern_raw.parquet      source (M1|M2), doc_id, split, text   (cache)
  data/interim/vuk_langid.parquet      text, label (eng|xho)                  (cache)
  data/processed/modern_lexicon.tsv    lexicon from LEXICON-role text only
  data/processed/modern_chunks.parquet sampled dev/test chunks (handbook D6.4)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.clean import normalise_unicode  # noqa: E402
from src.config import P, base_parser, load_config  # noqa: E402
from src.io_utils import get_logger, log_stage_stats, write_parquet  # noqa: E402
from src.lexicon import build_lexicon, save_lexicon  # noqa: E402
from src.modern_data import load_mzansi, load_vuk, peek_mzansi_langs  # noqa: E402
from src.segment import make_chunks  # noqa: E402
from src.split import sample_eval  # noqa: E402

# which original splits feed which role (handbook P5)
LEXICON_SPLITS = {"M1": {"train", "eval"}, "M2": {"validation"}}
DEV_SPLITS = {"M1": {"eval"}, "M2": {"validation"}}
TEST_SPLITS = {"M1": {"test"}, "M2": {"test"}}


def download(cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = cfg["modern"]
    vuk_x = load_vuk(m["vuk"]["hf_id"], m["vuk"]["config"], m["vuk"]["text_field"]).assign(source="M1")
    vuk_e = load_vuk(m["vuk"]["hf_id"], m["vuk"]["eng_config"], m["vuk"]["text_field"])
    log.info("Vuk'uzenzele xho rows %d, eng rows %d", len(vuk_x), len(vuk_e))
    frames = [vuk_x]
    for split in ("validation", "test"):
        df = load_mzansi(m["mzansi"]["hf_id"], split, m["mzansi"]["lang_value"], m["mzansi"]["max_rows"])
        log.info("MzansiText %s isiXhosa rows %d", split, len(df))
        frames.append(df.assign(source="M2"))
    raw = pd.concat(frames, ignore_index=True)[["source", "doc_id", "split", "text"]]
    langid = pd.concat([vuk_x[["text"]].assign(label="xho", split=vuk_x["split"]),
                        vuk_e[["text"]].assign(label="eng", split=vuk_e["split"])], ignore_index=True)
    return raw, langid


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--offline", action="store_true", help="use the cache only")
    parser.add_argument("--peek-mzansi", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p5_modern")
    if args.peek_mzansi:
        counts = peek_mzansi_langs(cfg["modern"]["mzansi"]["hf_id"])
        log.info("MzansiText lang values: %s", dict(counts))
        log.info("set modern.mzansi.lang_value in config.yaml to the isiXhosa code and log it")
        return

    raw_path, lid_path = P(cfg, "interim", "modern_raw.parquet"), P(cfg, "interim", "vuk_langid.parquet")
    if args.offline or args.smoke or (raw_path.exists() and lid_path.exists()):
        log.info("using cached modern data in %s", raw_path.parent)
        raw, langid = pd.read_parquet(raw_path), pd.read_parquet(lid_path)
    else:
        raw, langid = download(cfg, log)
        write_parquet(raw, raw_path)
        write_parquet(langid, lid_path)

    raw = raw.assign(text=raw["text"].map(lambda t: normalise_unicode(str(t)).replace("\n", " ")))
    lex_rows = raw[[s in LEXICON_SPLITS[src] for src, s in zip(raw["source"], raw["split"])]]
    lexicon = build_lexicon(lex_rows["text"], cfg["lexicon"]["min_freq"])
    save_lexicon(lexicon, P(cfg, "processed", "modern_lexicon.tsv", mkdir=True))
    log.info("lexicon: %d types from %d lexicon-role documents", len(lexicon), len(lex_rows))

    s = cfg["sample"]
    out = []
    for role, splits, per_source in (("dev", DEV_SPLITS, s["dev_per_source"]),
                                     ("test", TEST_SPLITS, s["test_per_source"])):
        docs = raw[[sp in splits[src] for src, sp in zip(raw["source"], raw["split"])]]
        blocks = [{"block_id": r.doc_id, "source": r.source, "edition_id": r.doc_id, "text": r.text}
                  for r in docs.itertuples()]
        ch = pd.DataFrame(make_chunks(blocks, cfg["segment"]["chunk_words"]))
        if ch.empty:
            continue
        ch["doc_id"] = ch["edition_id"]
        ch["sample_rank"] = sample_eval(ch, per_source, s["max_per_modern_doc"], "doc_id", seed=cfg["seed"])
        ch = ch[ch["sample_rank"] >= 0].assign(role=role, cluster_id=lambda d: d["doc_id"])
        out.append(ch)
    chunks = pd.concat(out, ignore_index=True)[
        ["chunk_id", "cluster_id", "source", "doc_id", "role", "sample_rank", "text", "n_words"]]
    write_parquet(chunks, P(cfg, "processed", "modern_chunks.parquet"))
    log_stage_stats(cfg, "P5 modern chunks", chunks.assign(edition_id=chunks["doc_id"]))
    log.info("modern chunks by source/role:\n%s", chunks.groupby(["source", "role"]).size().to_string())


if __name__ == "__main__":
    main()
