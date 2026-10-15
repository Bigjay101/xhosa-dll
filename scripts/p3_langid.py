"""P3 Fully automatic language filtering (Member A). Run after p5_modern.py (needs
the lexicon and the cached Vuk'uzenzele eng/xho texts).

Outputs:
  results/langid_eval.csv            held-out accuracy / F1 of the classifier
  data/processed/english_wordlist.txt
  data/interim/blocks_lang.jsonl     blocks + p_eng, lang, glotlid_label, glotlid_agree, eng_token_rate
Sanity check S3: English share by words close to 1.2% (Isigidimi) and 27.2% (Imvo).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import P, base_parser, load_config  # noqa: E402
from src.io_utils import append_sanity_row, get_logger, log_stage_stats, read_jsonl, write_jsonl  # noqa: E402
from src.langid import (build_english_wordlist, eng_token_rate, fit_langid, glotlid_labels,  # noqa: E402
                        kappa, load_glotlid, predict_eng_prob, text_to_blocks)
from src.lexicon import load_lexicon_file  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    log = get_logger(cfg, "p3_langid")
    lc = cfg["langid"]

    # P3.1 labelled training data: the Vuk'uzenzele config name is the label
    lid = pd.read_parquet(P(cfg, "interim", "vuk_langid.parquet"))
    texts, labels = [], []
    for r in lid.itertuples():
        for b in text_to_blocks(str(r.text), lc["block_words"]):
            texts.append(b)
            labels.append(r.label)
    x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=cfg["seed"],
                                              stratify=labels)
    model = fit_langid(x_tr, y_tr, seed=cfg["seed"])
    pred = model.predict(x_te)
    ev = {"n_train": len(x_tr), "n_heldout": len(x_te), "accuracy": accuracy_score(y_te, pred),
          "f1_eng": f1_score(y_te, pred, pos_label="eng")}
    pd.DataFrame([ev]).to_csv(P(cfg, "results", "langid_eval.csv", mkdir=True), index=False)
    log.info("langid held-out: %s", ev)

    # P3.4 apply to historical blocks
    blocks = read_jsonl(P(cfg, "interim", "blocks_dedup.jsonl"))
    p_eng = predict_eng_prob(model, [b["text"] for b in blocks])
    for b, p in zip(blocks, p_eng):
        b["p_eng"] = float(p)
        b["lang"] = "eng" if p >= lc["p_eng_threshold"] else "xho"

    # P3.3 independent second opinion (optional)
    glot = None if args.smoke else load_glotlid(lc["glotlid_repo"], lc["glotlid_file"])
    if glot is None:
        log.warning("GlotLID unavailable (or smoke run): P3.3 and X6.4 skipped -- log this")
        for b in blocks:
            b["glotlid_label"], b["glotlid_agree"] = None, None
    else:
        glabs = glotlid_labels(glot, [b["text"] for b in blocks])
        ours = [b["lang"] for b in blocks]
        for b, g in zip(blocks, glabs):
            b["glotlid_label"] = g
            b["glotlid_agree"] = (g == b["lang"]) if g in ("eng", "xho") else None
        mask = [g in ("eng", "xho") for g in glabs]
        k = kappa([o for o, m in zip(ours, mask) if m], [g for g, m in zip(glabs, mask) if m])
        log.info("GlotLID agreement: Cohen's kappa = %.3f on %d blocks", k, sum(mask))
        pd.DataFrame([{**ev, "glotlid_kappa": k}]).to_csv(P(cfg, "results", "langid_eval.csv"), index=False)

    # P3.5 English words inside isiXhosa blocks
    lexicon = load_lexicon_file(P(cfg, "processed", "modern_lexicon.tsv"))
    eng_words = build_english_wordlist(list(lid.loc[lid["label"] == "eng", "text"].astype(str)),
                                       lexicon, lc["english_wordlist_size"])
    P(cfg, "processed", "english_wordlist.txt").write_text("\n".join(sorted(eng_words)), encoding="utf-8")
    for b in blocks:
        b["eng_token_rate"] = eng_token_rate(b["text"], eng_words)
    write_jsonl(blocks, P(cfg, "interim", "blocks_lang.jsonl"))

    # S3: English share by words vs the published estimates
    df = pd.DataFrame(blocks)
    for source, expected in cfg["historical"]["expected_english_share"].items():
        g = df[df["source"] == source]
        if g.empty:
            continue
        share = float(np.sum(g["n_words"] * (g["lang"] == "eng")) / g["n_words"].sum())
        ok = abs(share - expected) <= 0.05
        status = "SKIP" if args.smoke else ("PASS" if ok else "WARN")
        detail = f"{source}: English share {share:.3f} vs published {expected:.3f}"
        append_sanity_row(cfg, "S3", status, detail)
        log.info("S3 %s %s", status, detail)
    log_stage_stats(cfg, "P3 isiXhosa blocks", df[df["lang"] == "xho"])


if __name__ == "__main__":
    main()
