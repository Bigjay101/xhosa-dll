# Doubly Low-Resource?

**Modern language models, historical isiXhosa and orthographic variation**  
NLP Project 2 — COMS4054A / COMS7062A / COMS7066A

**Authors:** Mivuyo Mpepo, Jordan Tshibumbu and Okwukwechukwu Mbajiorgu

## 1. Project overview

This project investigates two questions:

- **RQ1:** How does the prediction difficulty of historical isiXhosa compare with contemporary isiXhosa, and where is that difficulty concentrated?
- **RQ2:** To what extent does historical spelling variation contribute to the measured difference?

The experiments evaluate two pretrained masked language models—**XLM-R-base** and **AfroXLMR-base**—without fine-tuning. Text is scored using **word-normalised PLL-word-l2r pseudo-negative log-likelihood** (nats per word; lower means greater predictability). We compare historical newspaper excerpts with two modern isiXhosa corpora, examine word-level contributions to the prediction gap, and evaluate three spelling interventions:

1. **Historical normalisation (`H-norm`):** replace selected historical variants with modern forms.
2. **Modern historicisation (`M1-hist`, `M2-hist`):** introduce selected historical spellings into contemporary text.
3. **Matched random noise (`M1-noise`, `M2-noise`):** introduce random character edits at approximately comparable changed-word rates.

**Scope:** The submitted paper reports X2 (historical–modern gaps), X3 (word-level analysis), X4 (normalisation) and X5 (historicisation and noise). It does **not** report MzansiLM causal-language-model, optional adaptation, blind-normalisation, dose-response or full depth experiments.

> **Reproducibility status:** The commands below describe the implemented core workflow **from frozen processed inputs**. A new Git clone alone is insufficient: most processed datasets and all complete per-chunk score files are not tracked in Git. The original full inference runs have **not** been independently reproduced from a clean environment during the final review. Saved summary CSVs can be inspected without rerunning inference, but they do not substitute for complete per-chunk score files when recomputing the analysis.

**Run commands from the repository root** (the directory containing `config.yaml`). Stop and investigate if any command fails.

**Do not use the legacy `run_all.sh` as the core reproduction command.** That script contains unsupported steps and flags, including an unimplemented MzansiLM route and optional analyses that are not part of the submitted experiments. `START_HERE.md` describes an earlier development workflow; the commands below supersede it for the reported experiments.

## 2. Requirements and installation

Use **Python 3.10** (the checked environment used Python 3.10.11). Model downloads require internet access. A CUDA-enabled GPU is recommended for scoring; CPU inference is possible but much slower.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell prevents environment activation, use `.\.venv\Scripts\python.exe` in place of `python`; changing the PowerShell execution policy is not required.

### Linux/macOS (Bash)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

To inspect whether PyTorch can see your GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The scorer selects CUDA when available and otherwise CPU. You may explicitly pass `--device cuda` or `--device cpu` to `scripts/p8_score.py`. If necessary, install a PyTorch build compatible with your GPU using the [official installation instructions](https://pytorch.org/get-started/locally/).

The `requirements.txt` versions were checked in a local environment on **2 November 2026**, but transitive dependencies and the original model checkpoint revisions are not fully locked. To record the environment for **your** run:

```bash
python -m pip freeze > environment-run.txt
```

Do not commit your `.venv/` directory. GlotLID/fastText is relevant to optional raw-data preparation; it is not required to score already frozen processed data.

## 3. Required inputs and data access

### 3.1 Frozen inputs for the reported results

For an exact attempt at reproducing the published experiment, obtain the **original frozen processed-data bundle from the project authors** and restore the following paths:

```text
data/processed/hist_chunks.parquet
data/processed/modern_chunks.parquet
data/processed/modern_lexicon.tsv
data/processed/english_wordlist.txt
data/processed/rewrite_table_v1.tsv
configs/rules_v1.yaml
config.yaml
```

Also preserve the original `data/processed/splits.csv` and `MANIFEST.txt` for provenance or rebuilding earlier stages. The frozen rewrite table and configuration may be tracked in the repository, but other processed inputs can be Git-ignored. **Do not assume they are present after cloning.** If no frozen-data bundle has been supplied, the core inference workflow cannot yet be executed on the identical evaluated sample. The raw-data reconstruction route in [Section 9](#9-optional-reconstruction-from-raw-data) is not an exact substitute for the missing original split assignments.

Source datasets:

| Source | Role | Location |
|---|---|---|
| *Isigidimi sama-Xosa* | Historical newspaper | [UCT ZivaHub DOI: 10.25375/uct.22332271](https://doi.org/10.25375/uct.22332271) |
| *Imvo Zabantsundu* | Historical newspaper | [UCT ZivaHub DOI: 10.25375/uct.22332268](https://doi.org/10.25375/uct.22332268) |
| Vuk'uzenzele isiXhosa | Contemporary reference M1 | [Hugging Face: `dsfsi/vukuzenzele-monolingual`](https://huggingface.co/datasets/dsfsi/vukuzenzele-monolingual) |
| MzansiText | Contemporary reference M2 | [Hugging Face: `anrilombard/mzansi-text`](https://huggingface.co/datasets/anrilombard/mzansi-text) |

Check the relevant source/model cards for current licences, access terms and revisions before redistribution. This README identifies sources; it does **not** certify all licences.

### 3.2 Check the restored inputs

```bash
python -c "from src.data_api import load_condition, load_lexicon; print({c: len(load_condition(c)) for c in ['H-orig', 'H-dev', 'M1-orig', 'M2-orig']}); print('Lexicon types:', len(load_lexicon()))"
```

Expected counts **for the frozen experiment**:

| Input | Chunks |
|---|---:|
| `H-orig` | 1,780 (780 Isigidimi; 1,000 Imvo) |
| `H-dev` | 600 |
| `M1-orig` | 135 |
| `M2-orig` | 999 |
| Modern lexicon | 75,254 word types |

The historical evaluation sample spans **142 newspaper editions**; M1 covers **27 documents**, and M2 covers **763 documents**. The evaluated chunks contain **32 whitespace-delimited words**. The data API excludes one known corrupt M2 chunk consistently across conditions.

The frozen spelling resources contain **2,150 mappings** from accepted **ASP** and **BH** rule families. The stored dictionary uses a minimum mapping support of 10; the family-level validation threshold of 20 pairs is a separate criterion. When checking the rewrite-table file hash on Windows, note that CRLF conversion changes raw bytes relative to canonical LF line endings.

## 4. Generate the five transformed conditions

**Protect the original results first.** Back up `results/`, `figures/`, existing score files, the frozen processed files and `config.yaml`. Running P10 rewrites files in `data/conditions/`, while P11 can overwrite summary CSVs.

For a fresh inference rerun, use a **separate repository working copy** and configure separate output paths in that copy to avoid reusing an old incomplete checkpoint. In `config.yaml`, under `paths`, set:

```yaml
results: results/reproduction
scores: results/reproduction/scores
figures: figures/reproduction
```

Keep all other configuration entries unchanged. The separate paths apply to P8/P11 and the P10 family-rates output; P10 still writes transformed texts to the configured `data/conditions/` path.

Build the transformed conditions:

```bash
python scripts/p10_conditions.py
```

This command uses the **frozen rules**, estimates historicisation rates from `H-dev`, and generates `H-norm`, `M1-hist`, `M2-hist`, `M1-noise` and `M2-noise` in `data/conditions/`. It also saves `family_rates.csv` under the configured results directory. It **does not** score either model or retune the rules.

Expected changed-word counts from the original frozen inputs:

| Condition | Words changed | Share |
|---|---:|---:|
| `H-norm` | 3,322 / 56,960 | 5.832% |
| `M1-hist` | 247 / 4,320 | 5.718% |
| `M1-noise` | 214 / 4,320 | 4.954% |
| `M2-hist` | 1,594 / 31,968 | 4.986% |
| `M2-noise` | 1,484 / 31,968 | 4.642% |

Historicisation and noise are **approximately** rate matched; they do not have identical realised edit counts. The original word denominators and chunk IDs are preserved. The P10 script does not accept `--smoke` or `--config`.

## 5. Score both language models

Configured model identifiers:

| Key | Hugging Face model |
|---|---|
| `xlmr` | `FacebookAI/xlm-roberta-base` |
| `afro` | `Davlan/afro-xlmr-base` |

For each model, run the **eight** conditions `H-orig`, `M1-orig`, `M2-orig`, `H-norm`, `M1-hist`, `M2-hist`, `M1-noise` and `M2-noise` (16 model-condition runs in total).

### Windows PowerShell

```powershell
$conditions = 'H-orig','M1-orig','M2-orig','H-norm','M1-hist','M2-hist','M1-noise','M2-noise'
foreach ($model in 'xlmr','afro') {
    foreach ($condition in $conditions) {
        python scripts/p8_score.py --model $model --condition $condition --subset core
        if ($LASTEXITCODE -ne 0) { throw "Scoring failed: $model / $condition" }
    }
}
```

### Linux/macOS (Bash)

```bash
set -e
for model in xlmr afro; do
    for condition in H-orig M1-orig M2-orig H-norm M1-hist M2-hist M1-noise M2-noise; do
        python scripts/p8_score.py --model "$model" --condition "$condition" --subset core
    done
done
```

Scores are saved under the configured `paths.scores` directory, with names such as `xlmr__H-orig__core.parquet`. There should be **16 complete files**, with 1,780 rows per historical condition, 135 per M1 condition and 999 per M2 condition. The scorer checkpoints incrementally and resumes from existing chunk IDs; for accurate reproduction, **do not resume across different texts, settings or model checkpoint revisions**. Lower `--batch-size` (e.g. `--batch-size 8`) if necessary for GPU memory.

Optional small diagnostic (not a complete result):

```bash
python scripts/p8_score.py --model xlmr --condition M1-orig --subset core --device cpu --limit 1
```

**Note:** This diagnostic may append an additional row to an existing score checkpoint. Run it only in a scratch/reproduction copy, not against your preserved submission outputs. It is not a substitute for the 16 complete scores.

## 6. Verify coverage and regenerate statistics

**Do not regenerate P11 summaries from partial score files.** The analysis scripts can skip incomplete inputs, and rerunning them can replace saved final CSVs with partial results.

To check exact chunk-ID coverage, save the following code as `verify_score_coverage.py` in the repository root (or run it as a temporary script):

```python
from pathlib import Path
import pandas as pd
from src.config import P, load_config
from src.data_api import load_condition

cfg = load_config()
conditions = (
    "H-orig", "M1-orig", "M2-orig", "H-norm",
    "M1-hist", "M2-hist", "M1-noise", "M2-noise",
)

for model in ("xlmr", "afro"):
    for condition in conditions:
        path = P(cfg, "scores", f"{model}__{condition}__core.parquet")
        if not path.is_file():
            raise FileNotFoundError(f"Missing score file: {path}")
        scores = pd.read_parquet(path)
        expected = load_condition(condition, cfg=cfg)
        if scores["chunk_id"].duplicated().any():
            raise ValueError(f"Duplicate chunk IDs: {path}")
        if set(scores["chunk_id"]) != set(expected["chunk_id"]):
            raise ValueError(f"Incomplete or mismatched chunk IDs: {path}")
        print(model, condition, len(scores), "OK")

print("All 16 score files have complete chunk-ID coverage.")
```

Run:

```bash
python verify_score_coverage.py
```

Only **after this check succeeds**, regenerate the analysis:

```bash
python scripts/p11_stats.py --models xlmr afro
python scripts/p11_where.py --models xlmr afro
python scripts/p11_figures.py
```

The reported statistical configuration uses **seed 13** and **10,000 bootstrap/permutation draws**. `p11_stats.py` computes gap, normalisation, gap-closure, historicisation and noise estimates. `p11_where.py` performs word-category decomposition, regression and shared-word context analysis. `p11_figures.py` builds the condition-mean overview from complete score files; its optional family/dose plot is skipped when those unreported depth inputs are absent.

## 7. Results-to-paper mapping

The preserved `results/` summaries underpin the extended abstract:

| Report element | Output |
|---|---|
| Table 1 — historical–modern gaps | `results/X2_summary.csv` (`G`) |
| Table 1 — normalisation gap closure | `results/X4_summary.csv` (`GC`; multiply fraction by 100 for percent) |
| Figure 1 — intervention effects | X4 `delta` rows (negated for *after minus before*) and X5 M1 `eta`/`nu` rows |
| Absolute model scores | `results/X2_condition_means.csv` |
| Word-category decomposition | `results/X3_decomposition.csv` |
| Regression analysis | `results/X3_regression.csv` |
| Context analysis | `results/X3_context_gap.csv` (if generated) |
| Orthographic rule validation | `results/rule_validation.csv` |
| Rule selection | `results/rule_tuning.csv` and `configs/rules_v1.yaml` |
| Dataset counts | Frozen data loaded through `src/data_api.py` |

**Figure distinction:** `scripts/p11_figures.py` generates the **absolute condition-mean overview** (`fig1_main.pdf`). The manuscript's **orthographic intervention effect-size figure** is a separate plot derived from the X4/X5 summaries; its plotting source and final PDF must be included with the manuscript if you want others to regenerate that exact layout. **Do not claim that `p11_figures.py` produces the manuscript's effect-size plot.**

The eight-test Holm family comprises four historical–modern gaps, two normalisation benefits and two M1 historicisation effects. Bootstrap tail-area estimates stored as `0` do **not** imply mathematically zero probabilities. Word-category contributions are share weighted and sum across six mutually exclusive categories. An old `X2_gap_ci_draft.csv` is not the final Table 1 source.

## 8. Tests and checks

```bash
python -m pytest -q -m "not slow"
python -m pip check
```

In the **2 November 2026 local check**, the test suite reported **74 passed, 5 skipped**, and `pip check` reported no broken dependencies. That check used synthetic or small test fixtures and offline model tests; it did **not** download the two original pretrained checkpoints or reproduce the full inference runs. Skipped legacy tests are not passes.

If your environment produces different results, retain the test output and investigate failures; do not assume the historical 74-pass result applies to every platform or revision.

## 9. Optional reconstruction from raw data

The primary reproduction route is **Sections 3–6 using the original frozen processed inputs**. Rebuilding from raw documents is a separate operation and can change split assignments or resulting values. Use a fresh working copy, retain the original `data/processed/splits.csv`, and do not overwrite the frozen experimental artifacts.

Download the newspaper `.docx` files from the archive DOIs into:

```text
data/raw/isigidimi/
data/raw/imvo/
```

With raw files, the original split assignments, and the necessary modern-data network access, the documented preparation order is:

```bash
python scripts/p1_inventory.py
python scripts/p1_extract.py
python scripts/p1_check.py
python scripts/p2_clean_dedup.py
python scripts/p5_modern.py
python scripts/p3_langid.py
python scripts/p4_chunk_split.py
```

The subsequent rule-mining and validation steps are:

```bash
python scripts/p9_audit.py
python scripts/p9_mine.py
python scripts/p9_prepare_context.py
python scripts/p9_select_anchors.py
python scripts/p9_build_vectors.py
python scripts/p9_validate.py
python scripts/p9_tune.py
```

Then continue with Sections 4–6. This complete raw-to-results procedure was **not rerun in the final review**. Results may change with dataset or model revisions. Some scripts require external resources or local preparation and not every script accepts a `--smoke` flag. The legacy `run_all.sh --smoke` is not the current validation route.

## 10. Known limitations and responsible use

- **Missing tracked inputs:** A public Git repository does not necessarily include the frozen processed input bundle or original split assignment files. Contact the project authors for the original bundle if it is not supplied with the submission.
- **Partial saved inference:** The locally reviewed checkpoint set was incomplete (including only one row of `xlmr__M1-orig__core.parquet`). Therefore the original reported results cannot be recomputed from the saved summary CSVs alone; the full inference workflow must be rerun with complete inputs.
- **Original environment:** Exact checkpoint revision hashes and original end-to-end hardware/runtime records were not recovered. The pinned local requirements are not proof of bitwise-identical reproduction.
- **Scope:** The current report includes two masked language models and core experiments only; optional causal-LM, blind, family/dose and adaptation analyses are outside the reported results.
- **Orthographic rules:** The ASP/BH mappings were selected automatically at a family level. They are not exhaustive or individually verified by isiXhosa speakers, and historical spelling is not inherently an error.
- **Inference:** Pseudo-NLL is a predictability measure, not comprehension or downstream accuracy. Genre, period, OCR/extraction and filtering differences may contribute to the historical–modern gap.
- **Licensing:** Dataset/model licence terms and redistribution permissions should be checked from the original source records. Do not publish raw datasets, derived corpora or restricted assets without the appropriate rights.

The manuscript should include an accurate contribution statement, NeurIPS 2024 checklist, and Faculty of Science AI Ethics declaration. AI-assisted coding, analysis and writing must be disclosed truthfully in the required submission materials.

## 11. Repository outputs and submission

The paper, source code, `requirements.txt`, this README, relevant configuration files, frozen rule resources and verified summary CSVs form the core submission. Keep the final manuscript figure and its plotting source with the manuscript package where applicable.

If the course requires **`README.txt` specifically**, include a corresponding plain-text copy of these instructions in addition to `README.md`; the Markdown file alone may not satisfy a filename-specific requirement.

**Reproducibility claim:** The documented scripts describe how to rebuild the reported comparisons **when the original frozen input files and model resources are available**. The public code and verified summary tables by themselves do not establish a fully independent rerun of every reported inference result.
