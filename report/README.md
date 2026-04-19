# End-Sem Project Report

Word/Docs-style technical report, restructured to match a top-graded reference
report. Single column, navy-blue section headers, navy-blue table headers,
sans-serif body, 14 numbered sections, no academic bibliography.

## Build on Overleaf (recommended, 2 min)

1. Open https://www.overleaf.com/ and sign in.
2. **New Project → Upload Project** → drop `report.tex`.
3. Click **Recompile**. First build takes ~10 seconds.
4. **Download PDF**.

All packages used (`helvet`, `xcolor`, `colortbl`, `titlesec`, `tabularx`,
`longtable`, `hyperref`, `tikz`, `listings`, `graphicx`) ship with Overleaf
by default — no extra setup.

## Local compile

```bash
cd report
pdflatex report.tex
pdflatex report.tex    # run twice so cross-references resolve
```

## What's inside (14 sections)

1. Abstract
2. Dataset Summary
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis
5. Model Selection and Training
6. Model Evaluation
7. System Architecture (Technology Stack + Module Structure)
8. Agentic Workflow Explanation (AgentState schema, 5-step pipeline, ReAct chat)
9. RAG Pipeline Explanation (incl. the **deployment hazard finding**)
10. Guardrail Architecture
11. Deployment and User Interface (all 3 tabs documented)
12. Practical Implications
13. Conclusion
14. **Team Contributions** — two detailed paragraphs, Praanshu's work itemised

## Screenshots to add before final submission

The report has **six figure placeholders** (italic `[Figure placeholder: …]`
boxes). Before submitting, capture these PNGs and drop them into `report/`.
Then delete the placeholder paragraph and uncomment the `\includegraphics`
line directly above it.

| Placeholder | Filename expected | How to capture |
|---|---|---|
| §5.3 Feature Importance bar chart | `fig_feature_importance.png` | Run `model_brain.py` locally and add a matplotlib `plt.barh` of `tree.feature_importances_` (or grab from `genaicapstone.py` line 171-177). |
| §6.2 Confusion Matrix heatmap | `fig_confusion_matrix.png` | Add `metrics.confusion_matrix(y_test, y_pred)` to `model_brain.py` and plot with seaborn. |
| §7 System Architecture diagram | `fig_architecture.png` | Export the Mermaid diagram from the project README via https://mermaid.live/ (copy Mermaid, paste, export PNG). |
| §8.3 Chat screenshot | `fig_chat.png` | Open the live app → AI Care Coordinator → type a question → screenshot. |
| §11.1 Single Patient Analysis tab | `fig_single_patient.png` | Live app → Single Patient Analysis tab → fill a patient → Analyse → screenshot. |
| §11.2 Batch Analysis tab | `fig_batch.png` | Live app → Batch Analysis tab after uploading a CSV → screenshot of KPIs + chart. |
| §11.3 Structured Workflow card | `fig_workflow.png` | Live app → AI Care Coordinator → Run Full Care Workflow → expand Step 3 → screenshot. |

To insert a screenshot, find the placeholder in `report.tex` that looks like:

```latex
\textit{\textbf{Figure placeholder:} insert the system architecture
diagram here as \texttt{fig\_architecture.png}.}
```

and replace the whole paragraph with:

```latex
\begin{center}
  \includegraphics[width=0.95\textwidth]{fig_architecture.png}
\end{center}
```

## Reference style source

The structure and visual style follow a previously top-graded end-sem
report on a different topic (credit-risk scoring). All content and
technical details are specific to this project.
