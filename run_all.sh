#!/usr/bin/env bash
# Run the whole pipeline in handbook order.
#   bash run_all.sh            full run on the real data
#   bash run_all.sh --smoke    tiny end-to-end run on data/smoke (T16; < 10 min on CPU)
# Build smoke data first: python scripts/p0_make_synthetic.py (before M1)
#                      or python scripts/p7_make_smoke.py   (after M1, from real data)
set -euo pipefail
cd "$(dirname "$0")"
SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then SMOKE="--smoke"; fi
PY="${PYTHON:-python}"
run() { echo; echo "=== $* ==="; "$PY" "$@" $SMOKE; }

# ---- Member A (Jordan): foundation, P1-P7 ----------------------------------
run scripts/p1_inventory.py
run scripts/p1_extract.py
run scripts/p1_check.py
run scripts/p2_clean_dedup.py
run scripts/p5_modern.py                     # modern corpora + lexicon (needed by P3)
run scripts/p3_langid.py
run scripts/p4_chunk_split.py                # M1 data freeze on the real run
run scripts/p6_tok_metrics.py --allow-missing --condition H-orig --condition M1-orig --condition M2-orig
run scripts/p7_data_card.py

# ---- Member C: scoring, P8 ---------------------------------------------------
for MODEL in xlmr afro mzansi; do
  for COND in H-orig M1-orig M2-orig; do
    run scripts/p8_score.py --model "$MODEL" --condition "$COND" --subset core
  done
done

# ---- Member B: spelling rules and interventions, P9-P10 ----------------------
run scripts/p9_audit.py
run scripts/p9_mine.py
run scripts/p9_validate.py
run scripts/p9_tune.py                       # M2 rules freeze on the real run
run scripts/p10_conditions.py
run scripts/p6_tok_metrics.py --allow-missing --condition H-norm --condition M1-hist --condition M2-hist || \
  echo "(skipped: condition files not built yet)"

# ---- Member C: score new conditions, analysis, figures, P10.6-P12 ------------
for MODEL in xlmr afro mzansi; do
  for COND in H-norm M1-hist M2-hist M1-noise M2-noise; do
    run scripts/p8_score.py --model "$MODEL" --condition "$COND" --subset core
  done
done
run scripts/p8_score.py --depth-all          # every depth condition, xlmr and afro (P10.6)
run scripts/p11_stats.py
run scripts/p11_where.py
run scripts/p11_figures.py
run scripts/p12_adaptation.py                # optional (RQ3)

echo; echo "run_all finished${SMOKE:+ (smoke)}"
