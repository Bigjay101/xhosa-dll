# Handover checklist (Jordan -> B and C), handbook P7

Milestone **H** is complete when every box is ticked and B and C have posted
"handover accepted" in the group chat. Target: end of Monday 19 October.

## Data (P1-P6)
- [ ] S1-S4 logged in results/sanity_log.csv (PASS, or WARN with an explanation in decision_log.md)
- [ ] Every CHECK in handbook B4.5 owned by A resolved and logged
- [ ] data/processed/ complete; MANIFEST.txt committed (M1)
- [ ] data/processed/ copied to the shared Drive folder nlp-p2-shared/
- [ ] X1 run on H-orig, M1-orig, M2-orig; P6.2 vocabulary check logged

## Handover package (P7.1-P7.5)
- [ ] src/data_api.py works on the real data (T7 passes; `load_condition('H-orig')` returns the test sample)
- [ ] DATA_CARD.md generated and the "Gotchas" paragraph written
- [ ] Smoke kit built from real data (`python scripts/p7_make_smoke.py`) and `bash run_all.sh --smoke` passes
- [ ] requirements.txt pinned; README.txt data section complete
- [ ] Stub modules for B and C untouched and importable

## Early writing (P7.6)
- [ ] Background and related work (~140 words) in the paper project
- [ ] Data paragraph (~100 words) with real X0 counts
- [ ] paper/refs.bib complete; every C-marked reference confirmed
- [ ] NeurIPS checklist draft (items 1-7, 9-15)
- [ ] AI Ethics Statement template with Jordan's own declaration

## Handover call (P7.7, Sat 17 Oct, 30-45 min)
- [ ] Walked through DATA_CARD.md and decision_log.md
- [ ] B ran `pytest -q -m "not slow"` and `bash run_all.sh --smoke` on their machine
- [ ] C ran `pytest -q -m "not slow"` and `bash run_all.sh --smoke` on their machine
- [ ] Open CHECK items for B/C assigned
- [ ] Agreed: Jordan answers data questions asynchronously until end of Mon 19 Oct

Sign-off: B ________  C ________  date ________
