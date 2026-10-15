# Starter kit -- start here (Jordan / Member A)

This kit implements handbook phases P0-P7 (Member A) and contains stubs with the
agreed signatures for Members B and C. Tests T1-T7 pass on synthetic data; nothing
here has been run on the real newspapers yet -- that is Jordan's job.

## Jordan, in order
1. Push this folder to a private GitHub repo `nlp-p2-isixhosa`; add B and C (P0.1).
2. `pip install -r requirements.txt`; `pytest -q -m "not slow"` -> 23 passed, 8 skipped (P0.2).
3. `python scripts/p0_make_synthetic.py && bash run_all.sh --smoke` -> should finish (stubs print TODO).
   If `python` is not on your path: `PYTHON=python3 bash run_all.sh --smoke`.
   Tell B and C they can now develop against data/smoke/ (synthetic) until M1.
4. Put the real .docx files in data/raw/isigidimi/ and data/raw/imvo/ (P1.1).
5. `python scripts/p1_inventory.py` -> resolve the file-naming CHECK (P1.2).
6. `python scripts/p1_extract.py` -> dates resolved for > 98% of editions? (P1.4)
7. `python scripts/p1_check.py` -> S1, S2 (P1.5). Investigate any WARN.
8. `python scripts/p2_clean_dedup.py` (P2).
9. `python scripts/p5_modern.py --peek-mzansi`, set modern.mzansi.lang_value, then
   `python scripts/p5_modern.py` (P5; downloads once, then cached).
10. `python scripts/p3_langid.py` -> S3 (P3).
11. `python scripts/p4_chunk_split.py` -> S4, MANIFEST.txt -> commit -> announce M1 (P4).
12. `python scripts/p6_tok_metrics.py --check-vocab`, then
    `python scripts/p6_tok_metrics.py --condition H-orig --condition M1-orig --condition M2-orig` (P6).
13. `python scripts/p7_make_smoke.py && bash run_all.sh --smoke`; `python scripts/p7_data_card.py` (P7).
14. Early writing, handover call, HANDOVER_CHECKLIST.md -> milestone H (P7.6-P7.7).

Every script takes `--smoke` and logs to logs/. Record every CHECK resolution and any
change in decision_log.md.

## B and C
Your modules are in src/ as stubs raising NotImplementedError, with skipped test
stubs in tests/. Implement them with your LLM following handbook Part C, keeping the
signatures. Read data only through src/data_api.py.
