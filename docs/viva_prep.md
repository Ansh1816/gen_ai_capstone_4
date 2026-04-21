# Viva Preparation — Q&A Cheat Sheet

Short, defensible answers to the questions most likely to come up. Organised
by topic. Read the **one-minute pitch** first — it's the answer to the most
common opener *"Walk me through your project."*

---

## 0 · The one-minute pitch

> "It's a two-phase healthcare operations system. The first phase is a
> classical machine-learning model — a scikit-learn Decision Tree trained
> on 110{,}527 rows of the Kaggle *Medical Appointment No Shows* dataset —
> that predicts the probability a patient will miss their upcoming
> appointment.
>
> The second phase wraps that model inside a LangGraph-based agentic
> workflow that produces a grounded, citation-backed intervention plan.
> There are five stages: risk assessment, LLM risk reasoning, guideline
> retrieval from a ChromaDB vector store that holds the hospital's own
> policy documents, intervention planning, and a final structured report.
> A hardened system prompt makes the agent refuse to invent policies when
> retrieval is empty and requires every claim to carry a `[Source:
> filename]` tag.
>
> The whole thing is deployed to Streamlit Community Cloud with the
> knowledge base committed in-repo so the hosted agent is grounded from
> the first boot. The LLM is Groq's free-tier Llama 3.1 8B."

That's ~45 seconds when read aloud.

---

## 1 · Architecture

### Q. Walk me through the architecture.

Four layers top-to-bottom:

1. **Streamlit UI** — three tabs (Single Patient / Batch / AI Care Coordinator).
2. **ML Prediction** — scikit-learn Decision Tree (`model.pkl`) + StandardScaler (`scaler.pkl`).
3. **Agentic Layer (LangGraph)** — two complementary paths:
   - A **5-node StateGraph** for structured, auditable care reports (`agent.py`).
   - A **ReAct chat agent** for open-ended questions (`chatbot_agent.py`).
4. **RAG Layer** — five Markdown policy documents → `RecursiveCharacterTextSplitter` → MiniLM embeddings → ChromaDB (40 chunks).

External: Groq Cloud Llama 3.1 8B for LLM calls, fpdf2 for PDF export.

### Q. What happens when a user clicks "Run Full Care Workflow"?

1. The Streamlit button calls `agent.run_agent(patient_data)`.
2. LangGraph invokes **risk_assessment** — Decision Tree outputs a probability; thresholds at 0.30 and 0.55 map it to LOW / MEDIUM / HIGH.
3. **risk_reasoning** — an LLM call explains the patient's specific risk drivers.
4. **guideline_retrieval** — similarity search over ChromaDB returns the top-5 chunks, each tagged with its filename.
5. **intervention_plan** — the LLM, fed the Step-2 analysis and the Step-3 excerpts, produces a numbered plan with source tags.
6. **report_generation** — a final LLM pass compiles the full structured report with a disclaimer.

The Streamlit UI then renders each step in an expandable card so the grader can see the reasoning at every stage.

### Q. Why two agents instead of one?

Two different failure modes needed two different designs.

- **For a structured, auditable patient report** we want a strict, deterministic DAG — no surprises, every step always runs. That's the 5-node `StateGraph`.
- **For a human chatting in natural language** ("what's the SMS policy for medium-risk patients?") we want the agent to pick which tool to call and when — that's the ReAct loop.

They share the same model and the same ChromaDB, so there's no duplication at the data layer.

### Q. Why is the workflow linear? Why no conditional branches?

The real decisions (risk tier, retrieval empty, citation compliance) are
better enforced by the **system prompt** and by the **ML output**, not by
graph edges. Making them graph edges would add complexity without safety
benefit — and LangGraph's linear DAG gives us guaranteed ordering, which
matters for audit logs.

---

## 2 · Model choice

### Q. Why Decision Tree? Why not Logistic Regression / Random Forest / XGBoost?

**Logistic Regression** was the natural baseline and we actually compared against it (see `genaicapstone.py`). But appointment data has strong non-linear interactions — e.g. *long lead time AND prior no-show AND on scholarship* is much worse than any of those three alone. Logistic regression assumes an additive form and misses those.

**Random Forest / XGBoost** would almost certainly squeeze a bit more accuracy from the minority class. We didn't pick them because:
- **Interpretability** — a single Decision Tree gives feature importances we can show in the UI and cite in the report. A forest is harder to explain to a clinic administrator.
- **Inference cost** — this model is a *tool* called by an LLM agent. Decision Tree inference is ~microseconds, which keeps the agent snappy.
- **Deployability** — fits in the Streamlit Cloud free-tier memory budget (75 KB pickle) with plenty of headroom for the sentence-transformer and ChromaDB.

Given the project's emphasis is on the **agentic layer** (end-sem rubric is 35% agentic, with an explicit note that "traditional ML/DL alone is NOT sufficient"), we spent our complexity budget on the agent, not on a deeper classifier.

### Q. Why `max_depth=10`?

Shallow enough to avoid overfitting 100K+ rows, deep enough to capture feature interactions like *young + long lead + prior miss*. Empirically, test accuracy plateaus around depth 10 and falls past depth 15 due to overfitting. It's also what the original exploration notebook converged on.

### Q. What about class imbalance? ~80% show-up / 20% no-show.

Three responses:

1. **We didn't use `class_weight='balanced'`** because we treat the probability output, not the class label, as the primary artefact. The UI tiers the probability into LOW / MEDIUM / HIGH, so the threshold is explicit, not hidden inside the classifier.
2. **We report minority-class metrics** (precision, recall, F1 on the "no-show" class) in addition to accuracy — because a null model that always predicts "show up" already gets ~80%.
3. **Operationally**, the tier map is calibrated to the outreach guidelines in `attendance_management.md` — 0.30 triggers SMS, 0.55 triggers a phone call. The tier is what matters, not the raw class.

### Q. What features did you use?

Fourteen after preprocessing: age, gender (one-hot), `AwaitingTime`, SMS received, hypertension, diabetes, scholarship, disability level (one-hot into `Handicap_0..4`), and `Num_App_Missed`.

### Q. What is `AwaitingTime`?

The absolute difference in days between the scheduled day (when the
appointment was booked) and the appointment day (when it's meant to
happen). It proxies "forgetting risk" — patients who book weeks in advance
are more likely to lose track of the date.

### Q. What is `Num_App_Missed` and why is it the strongest feature?

For each patient, their cumulative count of prior no-shows, computed with
a groupby-shift-cumulative-sum so we don't leak the current row's label:

```python
df.groupby("PatientId")["No-show"]
  .transform(lambda x: x.shift().fillna(0).cumsum())
```

It's the single strongest predictor because past behaviour is the best
predictor of future behaviour. This is consistent with the published
no-show literature (Dantas et al., 2018).

### Q. Why did you drop `Neighbourhood`?

Two reasons:
1. **Cardinality** — 81 distinct neighbourhoods would blow up the feature space or require a target-encoding scheme that risks leakage.
2. **Ethics** — location is a strong proxy for race and socio-economic status; baking it in invites the model to discriminate. We'd rather capture the underlying effect through the `Scholarship` flag.

### Q. What does the model output — class or probability?

Probability. We call `predict_proba()[:,1]` to get the no-show probability in [0,1], and the UI/agent map that to a tier using 0.30 and 0.55 thresholds.

### Q. How did you evaluate it?

25% stratified holdout, `random_state=42`. We reported accuracy, precision
and recall on the minority class, F1, and confusion-matrix counts.
Accuracy sits in the 76–78% range — only marginally above the null
baseline of 80% — but that's expected for this imbalanced dataset, and
recall on the minority class (the operationally meaningful metric) is
what the tier mapping exploits.

---

## 3 · RAG

### Q. What is RAG and why did you use it?

**Retrieval-Augmented Generation.** Instead of relying on the LLM's
parametric memory (which can hallucinate hospital policies that don't
exist), the agent retrieves excerpts from a vector database of the
hospital's *actual* policy documents and feeds them to the LLM as
context. The LLM is then prompted to cite them by filename.

Two concrete wins:

1. The answer is grounded in the hospital's own rules, not in whatever the LLM picked up from training data.
2. It's auditable — every policy claim carries a `[Source: filename]` tag so the user can open the file and verify.

### Q. Why ChromaDB and not FAISS?

ChromaDB ships with persistence out of the box (a SQLite file), which
means we can commit the populated index to the repo and the hosted
Streamlit container has a ready-to-use knowledge base on first boot.
FAISS needs either a separate save/load step or a wrapper to persist. For
a 40-chunk corpus, both are fast enough — the deciding factor was
deployability.

### Q. Why `all-MiniLM-L6-v2` for embeddings?

384-dimensional, 22M parameters, small enough to run on CPU in a
Streamlit Cloud free-tier container, and widely benchmarked to produce
strong retrieval on short technical passages. A bigger model (e.g.
`bge-large`) would improve retrieval quality a few points but wouldn't
fit in the memory budget alongside sentence-transformers, ChromaDB,
Streamlit, and the agents.

### Q. Why chunk size 500 with overlap 80?

- **500 chars** is roughly 2-3 short paragraphs — big enough to carry a policy clause with its context, small enough that the LLM's working memory is focused.
- **80 char overlap** prevents a key sentence being chopped exactly at a chunk boundary.
- **Splitter separators** prioritise Markdown headings (`\n## `, `\n### `) first so chunks naturally align with policy sections.

### Q. Why top-k=5 for the workflow and top-k=3 for the chat?

The workflow needs broader context to build a multi-step plan, so 5 gives
it more material. The chat agent works in a single-turn window — 3 is
enough to answer one question well without flooding the LLM's context.

### Q. What happens if the retriever returns nothing?

The tool returns the literal string `"No specific guidelines found for
this query."` The system prompt then requires the agent to say so
plainly — it is forbidden to invent a policy. This is one of our five
guardrails.

### Q. How do you evaluate the retrieval quality?

Qualitatively for now — when asked to quote a specific document, the
agent does quote it verbatim (verified live). A formal evaluation with a
held-out query-to-chunk labelled set (hit@k, nDCG) is future work.

---

## 4 · LangGraph

### Q. What is LangGraph and why did you use it?

LangGraph is a framework for building **stateful, graph-based** agent
workflows on top of LangChain. You define nodes (Python functions), edges
(transitions), and a shared **state** object that each node reads from
and writes to.

We used it because:
1. **Explicit state** — the `AgentState` `TypedDict` is a clear contract between nodes. No hidden globals.
2. **Determinism** — nodes fire in declared order. For an auditable medical-operations system, that's essential.
3. **Introspection** — we can expose each intermediate state to the UI, which is exactly what the *Run Full Care Workflow* button does.

### Q. What's `AgentState`?

A `TypedDict` carrying the shared memory across nodes:

```python
class AgentState(TypedDict):
    patient_data: dict
    risk_score: float
    risk_tier: str
    risk_color: str
    risk_analysis: str
    risk_factors: list
    retrieved_guidelines: list  # [{content, source}, ...]
    intervention_plan: str
    final_report: str
```

Each node returns a *partial* dict; LangGraph merges it into the state.

### Q. What's the difference between StateGraph and `create_react_agent`?

- **StateGraph** — you declare the DAG yourself. Good for deterministic, multi-step pipelines.
- **`create_react_agent`** — a prebuilt ReAct loop (Reasoning + Acting). The agent picks which tool to call based on the user's message, loops until it has an answer. Good for open-ended chat.

We use both: StateGraph for the 5-step care report, ReAct for the chat tab.

---

## 5 · LLM / Prompt engineering

### Q. Why Groq and why Llama 3.1 8B?

- **Groq** — fastest inference on open-weight models by a wide margin, and the free tier (30 requests/minute) is sufficient for demos and grading. No card required.
- **Llama 3.1 8B Instant** — strong enough to follow a tool-using system prompt and produce structured output, fast enough that a 5-step workflow finishes in 30–60 seconds. We tested 70B as well — quality improved marginally, latency tripled, free-tier quota drops.

### Q. Why temperature 0.3?

Low enough that the structured-output format stays stable across runs (JSON-like headings, numbered lists, source tags). High enough to get varied, patient-specific narratives rather than boilerplate.

### Q. What is a system prompt and how did you harden yours?

A system prompt is the set of instructions the LLM gets at the *start*
of every conversation — before the user's message. It defines the
agent's role and behaviour.

Ours was originally three generic sentences. We rewrote it to enforce
four hard rules:

1. **Always call `predict_noshow` first** for any patient question; report the exact percentage alongside the tier.
2. **Every policy claim must carry `[Source: <filename>]`.** If the excerpt didn't come from the `search_guidelines` tool, don't make the claim.
3. **If `search_guidelines` returns "No specific guidelines found"**, tell the user plainly. Never invent policies. Never fall back to generic healthcare advice.
4. **Always append the operational & ethical disclaimer** at the end of every structured output.

This one change moved the agent from fluent hallucinator to grounded assistant.

### Q. What's ReAct?

**Re**ason + **Act**. A prompting pattern where the LLM alternates between
"thought" steps (reasoning about what to do next) and "action" steps
(calling a tool). LangChain's `create_react_agent` is the prebuilt
implementation.

### Q. How does tool calling actually work?

The LLM is given a list of tools as structured JSON schemas (name,
description, arguments). When the agent decides to use a tool, it emits a
JSON object that matches one of the schemas. Our Python runtime
intercepts it, calls the actual Python function
(`predict_noshow` or `search_guidelines`), and feeds the result back to
the LLM as a tool-message in the next turn.

---

## 6 · Deployment

### Q. How is the app deployed?

Streamlit Community Cloud watches the `main` branch on GitHub. Every push
triggers a rebuild: the container is re-created, dependencies
reinstalled from `requirements.txt`, and the app starts with
`streamlit run app.py`.

`GROQ_API_KEY` is configured in the platform's Secrets panel.

### Q. Why did you commit `chroma_db/` to the repo?

Because Streamlit Cloud deploys only what is in the Git repo. If we had
left `chroma_db/` in `.gitignore`, the hosted container would boot with
an empty ChromaDB — the retriever would always return nothing, and the
agent's empty-retrieval fallback would fire on every query. We actually
hit this in live testing: the hosted agent was hallucinating because the
vector store wasn't shipping. Un-ignoring and committing the populated
store fixed it.

### Q. What about model artefacts?

`model.pkl`, `scaler.pkl`, and `feature_cols.pkl` are also committed so
the container doesn't need access to the training dataset. The Kaggle
CSV (`KaggleV2-May-2016.csv`) is *not* committed — it's large, and the
app doesn't need it at runtime.

---

## 7 · Ethics & Safety

### Q. How do you handle ethics?

Five enforcement layers:

1. **Human-in-the-loop** — every structured output ends with an explicit disclaimer that recommendations must be reviewed by qualified administrative staff before action.
2. **Source attribution** — the UI shows source-filename chips beneath every retrieval result, so grounding is visible.
3. **Hallucination refusal** — the system prompt forbids inventing policies when retrieval is empty.
4. **No medical advice** — the disclaimer explicitly states the system is operational decision support, not clinical advice.
5. **Data handling** — patient data goes only to Groq for LLM completion; nothing is persisted on disk. Production use would require HIPAA/GDPR controls.

### Q. What about fairness?

`Neighbourhood` was deliberately dropped during preprocessing to avoid
encoding a proxy for race/income. We did keep `Scholarship` (a
socioeconomic signal) because it's directly interpretable and covered by
the intervention guidelines in `patient_engagement.md`. Our
`ethical_guidelines.md` document spells out that regular bias audits
across demographic groups should be part of the production rollout.

---

## 8 · Metrics

### Q. Why is recall on the no-show class so low (~0.25–0.35)?

Because the dataset is heavily imbalanced and the patterns that drive
no-shows are genuinely subtle — patients who miss appointments don't
always differ in measurable ways from patients who attend. Published
baselines on the same dataset sit in the same range, so we're not
underperforming the field.

Three things would improve recall:
1. **Class-weighted training** (e.g. `class_weight='balanced'`).
2. **Threshold calibration** — lower the 0.55 threshold on the minority class.
3. **Better features** — day-of-week and time-of-day, which are in the raw data but we didn't engineer.

### Q. Why not just report accuracy?

Because a null classifier that always predicts "shows up" already hits
~80%. Accuracy is misleading on imbalanced data. Minority-class
precision and recall are the operationally meaningful metrics.

---

## 9 · "Why not X?" curveballs

| Question | Short answer |
|---|---|
| *Why not OpenAI GPT-4?* | Cost. The brief restricts us to free-tier APIs. Groq Llama 3.1 is free and fast. |
| *Why not a larger model (70B)?* | Latency tripled and free-tier quota drops. Quality gain is marginal for this task. |
| *Why not use LlamaIndex?* | LangGraph gave us explicit state, which the rubric's "explicit state management" bullet rewards. |
| *Why not FAISS?* | ChromaDB persists as a SQLite file so we can ship a pre-built index in the repo. |
| *Why not fine-tune the LLM?* | Free-tier hosted models don't expose fine-tuning; RAG gives us the grounding we need without a training run. |
| *Why didn't you use XGBoost / a stronger classifier?* | Interpretability + inference cost matter more than 2 extra points of accuracy. The grader's focus is the agentic layer. |
| *Why Streamlit over Gradio?* | Custom CSS and multi-tab layout were easier in Streamlit; the rubric accepts either. |
| *Why not automate `build_vectorstore` on startup?* | Adds ~30–60 s to every cold start for no benefit once the index is committed. |

---

## 10 · If you don't know the answer

Stay calm. Three safe moves:

1. **"Good question — let me think for a second."** Then reason out loud. They're often evaluating your reasoning more than the final answer.
2. **Say what you *do* know and what you'd *check*.** *"I'd verify the exact number, but the order of magnitude is X, and the reason is Y."*
3. **Tie back to what's in the report.** *"Section 9 of the report covers this — we documented the failure mode and the fix."*

What not to do:
- Don't make up numbers. If you're unsure, say so.
- Don't guess prompts or code. If they ask "what exactly does your system prompt say?", offer to open the file.

---

## 11 · Thirty-second closers

If they ask *"What was the hardest problem?"*:

> "Making RAG actually work in production. Our first deploy had the ChromaDB folder excluded by `.gitignore`, so the hosted agent was running with an empty vector store and hallucinating hospital policies that looked real. We caught it with a specific quote-test — asking the agent to copy three bullets from a document verbatim — and it failed cleanly. The fix was to un-ignore the folder, commit the populated index, and harden the system prompt to forbid inventing policies on empty retrieval. That story is in Section 9 of the report."

If they ask *"What would you do with more time?"*:

> "Three things. One — formal retrieval evaluation, hit@k and nDCG on a labelled query set. Two — a class-weighted XGBoost baseline, just to confirm the interpretability trade-off was worth the recall lost. Three — more corpus coverage; five policy files demonstrate the idea, but a production rollout would need the full operations manual indexed with a freshness check."

If they ask *"What did you learn?"*:

> "That the hardest part of an agentic system isn't the agent — it's the infrastructure around it. The LangGraph code was straightforward. The hard parts were getting the vector store into production, writing prompts that refuse to hallucinate, and making the UI transparent enough that a grader can see every step of the reasoning."
