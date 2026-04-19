# End-Sem Project Report

Word/Docs-style 23-page technical report. Single-column, navy-blue section
headers and table headers, sans-serif body, 14 numbered sections, no
academic bibliography. Styled to match a previously top-graded reference
report.

## Ready-to-submit PDF

[`report.pdf`](report.pdf) — 741 KiB, 23 A4 pages, fully compiled with all
figures and screenshots embedded. This is the file you submit.

## What's inside

1. Abstract
2. Dataset Summary (attribute + feature tables)
3. Data Cleaning and Preprocessing (age, encoding, scaling, derived features)
4. Exploratory Data Analysis (key observations + visual insights)
5. Model Selection and Training — **includes real feature-importance bar chart**
6. Model Evaluation — **real holdout metrics (Accuracy 79.88%, Precision 0.51, Recall 0.07, F1 0.13, ROC-AUC 0.73) and real confusion-matrix heatmap**
7. System Architecture — **includes matplotlib architecture diagram**
8. Agentic Workflow Explanation (AgentState schema, 5-step pipeline, ReAct agent)
9. RAG Pipeline Explanation (incl. the deployment hazard finding)
10. Guardrail Architecture (5-layer table)
11. Deployment and User Interface — **includes 4 live-app screenshots** (single patient, batch, chat, structured workflow)
12. Limitations and Future Work
13. Conclusion
14. **Team Contributions** — two detailed paragraphs, Praanshu's work itemised

## Figures (all embedded in `report.pdf`)

| File | Source |
|---|---|
| `fig_architecture.png` | Matplotlib-rendered 4-layer system architecture diagram |
| `fig_feature_importance.png` | Bar chart generated directly from the persisted `model.pkl`'s `feature_importances_` |
| `fig_confusion_matrix.png` | Real 2×2 matrix from retraining on the Kaggle V2 25%-holdout split |
| `fig_single_patient.jpg` | Live capture of the Single Patient Analysis tab |
| `fig_batch.jpg` | Live capture of the Batch Analysis tab |
| `fig_chat.jpg` | Live capture of the AI Care Coordinator chat with a policy question and a source-cited answer |
| `fig_workflow.jpg` | Live capture of the Run Full Care Workflow structured output with Step 3 expanded |

Plus [`metrics.json`](metrics.json) with the exact holdout metrics computed
during the run.

## Rebuild the PDF

### Overleaf (easiest)

Upload `report.tex` plus all `fig_*.png` and `fig_*.jpg` files to an
Overleaf project → Recompile → Download PDF.

### Local with Tectonic (recommended)

```bash
brew install tectonic              # one time
cd report
tectonic report.tex                # ~15 s, pulls packages on demand
```

### Local with TeX Live

```bash
cd report
pdflatex report.tex
pdflatex report.tex                # second pass to resolve cross-refs
```

## Regenerate the charts

If you edit the ML model, re-run the chart generator:

```bash
source ../venv/bin/activate
python3 /tmp/gen_figures.py        # rebuilds fig_architecture, fig_feature_importance, fig_confusion_matrix
```

(The script lives at `/tmp/gen_figures.py` — copy to the repo if you want
to version-control it.)

## Regenerate screenshots

The four app screenshots were captured via an automated browser run against
the live Streamlit app. If the UI changes, re-capture by screenshotting the
relevant tab manually and overwriting the corresponding `fig_*.jpg`.
