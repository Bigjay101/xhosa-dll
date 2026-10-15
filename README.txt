DOUBLY LOW-RESOURCE: modern language models and 19th-century isiXhosa newspapers
NLP Project 2 (COMS4054A / COMS7062A / COMS7066A) - Jordan [surname], [Name B], [Name C]
(Draft by Member A, handbook P7.5. B completes the analysis sections in P14.2.)

1. REQUIREMENTS
   Python 3.10 or 3.11. A GPU is recommended for model scoring (Colab works).
     python -m venv .venv && source .venv/bin/activate
     pip install -r requirements.txt

2. DATA
   Historical newspapers (public domain), from the isiXhosa Intellectual Traditions
   archive on UCT ZivaHub:
     Isigidimi sama-Xosa  doi:10.25375/uct.22332271  -> data/raw/isigidimi/
     Imvo Zabantsundu     doi:10.25375/uct.22332268  -> data/raw/imvo/
   Modern isiXhosa corpora are downloaded automatically from Hugging Face on the
   first run of scripts/p5_modern.py and cached in data/interim/ (internet needed once):
     dsfsi/vukuzenzele-monolingual (CC BY 4.0), anrilombard/mzansi-text (Apache-2.0).

3. RUN
     bash run_all.sh            full pipeline in handbook order; approx. [X] h on [GPU]
     bash run_all.sh --smoke    tiny end-to-end check on data/smoke (< 10 min on CPU)
   Build smoke data first:
     python scripts/p0_make_synthetic.py   (synthetic, before the real data exists)
     python scripts/p7_make_smoke.py       (from the real data, after the data freeze)
   Individual steps: python scripts/pN_*.py [--smoke]   (order: see run_all.sh)

4. TESTS
     pytest -q -m "not slow"    fast tests (no downloads)
     pytest -q                  everything (needs model downloads / GPU)

5. OUTPUTS
     results/*.csv     every number in the paper (results_log.csv indexes them)
     results/sanity_log.csv   sanity checks S1-S8
     figures/*.pdf     figures in the paper
     logs/             one log per script run, with config hash and git commit
     DATA_CARD.md      description of the processed data

6. RESULTS-TO-PAPER MAPPING
     Table 1  -> results/X1_summary.csv (+ X3 explained share, context gap)
     Fig. 1   -> results/X2_summary.csv, X4_summary.csv, X5_summary.csv
     Fig. 2   -> results/X4_summary.csv, X5_summary.csv, X5_dose.csv
     Supplementary -> X0_data_stats.csv, X3_*.csv, X6_sensitivity.csv,
                      X7_summary.csv (optional), rule_validation.csv

7. SEEDS        seed 13 everywhere (config.yaml)
8. AI USE       see the AI Ethics Statement in the report
