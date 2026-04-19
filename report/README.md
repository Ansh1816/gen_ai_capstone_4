# Project Report (LaTeX)

IEEE-conference-style end-semester report for the **Clinical Appointment
No-Show Prediction and Agentic Care Coordination** capstone.

## Build options

### Option A — Overleaf (recommended, no local TeX install)

1. Go to https://www.overleaf.com/ and sign in.
2. Click **New Project → Upload Project**.
3. Upload `report.tex`.
4. Overleaf will auto-detect it as a LaTeX project. Click **Recompile**.
5. Download the produced PDF.

Overleaf ships `IEEEtran`, `tikz`, `booktabs`, `tabularx`, `listings`, and
`hyperref` out of the box — no extra packages to install.

### Option B — Local compile with TeX Live

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # run twice so references resolve
```

The report compiles cleanly with a stock TeX Live install. If `pdflatex`
reports missing packages, install the `texlive-publishers` bundle (which
provides `IEEEtran`) and `texlive-pictures` (for TikZ).

### Option C — Docker

```bash
docker run --rm -v "$PWD":/doc -w /doc \
  texlive/texlive:latest pdflatex report.tex
```

## What's inside

- Title, authors, affiliations (Newton School of Technology)
- Abstract + IEEE keywords
- 10 sections: Introduction → Related Work → System Architecture → Milestone 1
  (ML) → Milestone 2 (Agentic) → RAG → Implementation & Deployment →
  Qualitative Case Study → Ethical Safeguards → Results & Discussion →
  Limitations → Conclusion
- Two TikZ diagrams (system architecture + 5-step LangGraph workflow)
- Tables for the feature set and ML metrics
- Code listings (state TypedDict, hardened system prompt, cumulative
  prior-miss feature)
- Explicit **Team Contributions** section naming both authors
- 12 references

## Screenshot placeholder

Section IX (Qualitative Case Study) references `Fig. 3 — structured workflow
output, live on Streamlit Community Cloud` as a placeholder. Before final
submission, take a screenshot of the deployed app's 5-step workflow card
(with Step 3 expanded, showing the source chips) and replace the placeholder
`\fbox{\parbox{...}}` with:

```latex
\includegraphics[width=\columnwidth]{workflow_screenshot.png}
```

Place `workflow_screenshot.png` in the same `report/` folder.
