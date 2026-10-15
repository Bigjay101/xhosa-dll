# Decision log (handbook D6.5)

Every deviation from the handbook, every CHECK resolution and every scope decision.

| date | step | change | reason | effect on results | who |
|---|---|---|---|---|---|
| 2026-10-14 | all | Handbook v2 adopted: RQ1 + RQ2 in depth, RQ3 optional, time/newspaper RQ and LAFT dropped | Lecturer feedback | Fewer experiments, deeper analysis | group |

2026-10-15  P1 extraction (Jordan)
- Archive TXT export is one file per page; pages grouped into editions by file-name date
  (201 Isigidimi, 517 Imvo vs published 202/518; 0 unresolved dates).
- S1: Isigidimi 0.98 PASS. Imvo 0.73 WARN, explained: English function-word share 0.7%
  -> English columns appear not to be transcribed; published count likely covers the
  whole bilingual paper. Shortfall is uniform (~730 words/page), not broken files.
  [confirm against ZivaHub README]
- Consequence: English blocks already absent upstream; code-switching analysis is limited
  to English insertions inside isiXhosa text.
- S2: per-year flags (Isigidimi 1874, 1881) reflect issues missing from the archive
  (history model implies ~222 issues; archive has 202).
- Kept as-is: Isigidimi 1875-01..03 (16-19 pages; p18 missing in March),
  1870-10-14 (2 pages), Imvo 1893-10-18 (p4 only).
- Licence: check ZivaHub record; UCT lists converted text as CC BY-NC-ND. Text never committed.

2026-10-15  P2 dedup (Jordan)
- Removed: Imvo 18.2% of words (97.6k exact + 1.2k near blocks), Isigidimi 4.7%.
- Profile (p2_diagnose): median removed block 5 words; 84% of removed Imvo blocks <10 words;
  median recurrence span 42 days; 3% of copies within the same edition; 79 texts recur 100+ times
  -> repeated adverts, notices, mastheads. Kept earliest copy.
- Effect on evaluation is small: blocks <32 words never form chunks; at most ~4% of Imvo words
  (blocks of 20+ words) could have reached chunking.
- Archive duplicates found: IMVO_1885-03-02 p002 = p004; IMVO_1886-01-06_p002 = IMVO_1886-01-20_p003.
  The real text of one page in each pair is missing from the archive; the copies are removed by dedup.
- Corpus descriptions (word totals, S1) use pre-dedup counts; all frequency and model results use deduped data.

2026-10-15  P5 modern data (Jordan)
- Vuk'uzenzele text column = `text` (P5.1); MzansiText isiXhosa lang value = `xho` (P5.2).
- Vuk'uzenzele: 181 xho / 150 eng articles. MzansiText xho: 3,452 validation / 3,296 test.
- Lexicon: 75,254 types (Vuk train+eval, MzansiText validation; min_freq 2). No test docs.
- Modern chunks: M1 130 dev / 135 test (capped by 28 test articles x 5 chunks; wider CIs for M1),
  M2 300 dev / 1,000 test.

2026-10-15  P3 language filter (Jordan)
- Classifier (char 1-4-gram TF-IDF + LR, Vuk'uzenzele eng/xho): held-out acc 0.999 on modern text.
- On the newspapers it flagged 24,167 blocks (~5% of words); flagged blocks were capitalisation-
  and digit-heavy (name lists, prices, notices) with few English words. GlotLID called only 5.3%
  of flagged 32+-word blocks English.
- English word list fixed: function words were being removed by overlap with the web-crawled
  isiXhosa lexicon; now kept unless >=5x relatively more frequent in isiXhosa (1,866 words, 10/10 core).
- Rule: English = p_eng >= 0.5 AND >= 50% common-English words (mostly-English blocks only;
  mixed blocks kept for the code-switching analysis). 334 blocks; 0.04% (Imvo) / 0.02% (Isigidimi) of words.
- Robustness: a 0.2 cut-off instead of 0.5 changes retained words by < 0.1% per newspaper.
- S3 WARN for Imvo explained: English columns were not in the TXT export (see P1).
- GlotLID kappa (0.08) not reported: unstable when English is < 0.1% of blocks.