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

---

### 1.1 · End-to-end request flows (diagram walk-through)

Every flow below is a walk across the system architecture diagram (see
`docs/architecture.md`). Read each one while pointing at the diagram —
you are literally tracing the arrows.

**Diagram nodes referenced (by name and colour):**

- **Streamlit UI** (blue) — the three-tab front-end.
- **DecisionTreeClassifier** (slate, ML Layer) — the trained no-show model.
- **ReAct Agent** (teal) — the conversational chat agent.
- **5-Step Care Workflow pod** — risk_assessment → risk_tier gate →
  risk_reasoning → guideline_retrieval → retrieval-empty gate →
  intervention_plan → carries-Source-tag gate → report_generation →
  structured output.
- **RAG Pipeline pod** — guidelines → chunker → embedder → ChromaDB.
- **Groq Cloud** (purple dashed) — external LLM.
- **pdf_generator** (slate) — PDF export.

---

#### Flow A — User clicks "Analyse Appointment Risk" (Tab 1, Single Patient)

Trace one arrow on the diagram: **Streamlit UI → DecisionTreeClassifier**.

1. The user fills the patient form on the left of the Streamlit UI.
2. On click, the UI sends the form values along the *"Single Patient /
   Batch tab"* arrow into the **ML Prediction Layer**.
3. The DecisionTreeClassifier returns a single no-show probability.
4. The UI maps that probability into a **risk tier** (LOW / MEDIUM /
   HIGH) using the 0.30 and 0.55 thresholds and renders the result
   cards — risk percentage, tier badge, rule-based recommendation,
   what-if simulator.

The arrow is a **single hop**, in and back out. The diagram nodes that
are *not* touched in this flow: the ReAct Agent, the Workflow pod, the
RAG Pipeline, Groq, the PDF generator. **No LLM, no RAG, no agent.**

---

#### Flow B — User uploads a CSV (Tab 2, Batch Analysis)

Same arrow as Flow A — **UI → DecisionTreeClassifier** — just with many
rows instead of one.

1. The CSV lands in the Streamlit UI.
2. The UI passes every row along the *"Single Patient / Batch tab"*
   arrow into the ML layer.
3. The DecisionTreeClassifier returns one probability per appointment.
4. The UI renders the KPI row, the risk-distribution bar chart, the
   operational-actions panel with headcounts per tier, and a sortable
   high-risk patient list.

Again, only two architecture layers participate — **UI and ML**. The
agent and RAG layers are untouched.

---

#### Flow C — User types in the AI Care Coordinator chat

This flow activates the **teal ReAct Agent** node and its two outgoing
tool arrows.

1. The user's message enters the **UI** and follows the *"Chat
   message"* arrow into the **ReAct Agent**.
2. The ReAct Agent starts its **reasoning loop** along the dashed
   arrow to **Groq Cloud**. The LLM reads the hardened system prompt
   plus the conversation history and decides what to do next.
3. The LLM either answers directly, or it calls one of the agent's
   two tools:
   - *"predict_noshow tool"* arrow → back to the **ML layer**. The
     Decision Tree returns a probability and tier, which the tool
     hands back to the LLM as context.
   - *"search_guidelines tool"* arrow → across to **ChromaDB** in the
     RAG pod. The vector store returns matching excerpts with
     `[Source: filename]` tags (or an explicit empty-fallback string
     if nothing matches).
4. The LLM incorporates the tool output, loops again if needed, and
   eventually produces a final answer.
5. The answer travels back up to the **UI** and is rendered in the
   chat panel with the source filenames visible.

In architecture terms: **UI → ReAct Agent → (Groq + ML + ChromaDB)
→ UI.** The diagram's gates inside the Workflow pod are not used here.

---

#### Flow D — User clicks "Run Full Care Workflow"

This is the showpiece. Trace it through the **Workflow pod** of the
diagram, one node at a time, with side-trips to Groq and ChromaDB.

1. **Entry.** The user's patient profile travels along the *"Run Full
   Care Workflow"* arrow from the **UI** into the Workflow pod.
2. **risk_assessment (slate node, Step 1).** The incoming patient data
   is routed to the **ML layer** (dashed arrow *"predict"*). The
   Decision Tree returns the no-show probability, which the node
   attaches to the shared workflow state.
3. **risk_tier gate (amber, Step 1-to-2).** The probability passes
   through the amber diamond. Thresholds 0.30 and 0.55 classify it as
   LOW, MEDIUM, or HIGH. The tier label joins the state.
4. **risk_reasoning (teal, Step 2).** The node follows the dashed
   arrow to **Groq Cloud**. The LLM receives the patient values, the
   probability, and the tier, and returns a natural-language
   explanation of *why* this patient is at this tier.
5. **guideline_retrieval (slate, Step 3).** The node composes a query
   from the tier and the top risk factors. It follows the *"top-k=5
   similarity"* arrow into **ChromaDB** (RAG pod). The vector store
   returns five policy excerpts, each carrying its `[Source:
   filename]` metadata. The node attaches them to the state.
6. **retrieval-empty gate (amber, Step 3-to-4).** If ChromaDB returned
   zero matches, the flow follows the dashed **"zero docs"** branch
   to the *"inform user — no invented policy"* node and stops. On the
   happy path, the *"docs found"* solid arrow continues to
   intervention_plan.
7. **intervention_plan (teal, Step 4).** The node makes another dashed
   call to **Groq Cloud**. The LLM is fed the Step-2 risk analysis and
   the Step-3 excerpts and produces a structured plan: primary action,
   numbered strategies, timeline, escalation protocol, expected
   outcome. Every policy claim in the plan carries the source tag it
   was grounded in.
8. **carries-Source-tag gate (amber, Step 4-to-5).** A second amber
   diamond checks that the plan actually contains the required
   citation tags. If it does, the solid *"yes"* arrow proceeds to
   report_generation. (If a future version detects missing tags, the
   dashed *"no"* loop re-asks the plan step.)
9. **report_generation (teal, Step 5).** Third dashed call to **Groq
   Cloud**. The LLM compiles the final structured care-coordination
   report from the patient summary, the risk, the reasoning, the plan,
   the sources, and the mandatory operational-and-ethical disclaimer.
10. **structured output (slate, END of the pod).** The complete state
    exits the Workflow pod and returns to the **Streamlit UI**, which
    renders each of the five steps as an expandable card — including
    the source-filename chips for Step 3 — plus a permanent
    ethical-notice banner.
11. **Optional export.** If the user clicks the PDF button, the solid
    arrow continues from *structured output* to **pdf_generator**,
    which produces a downloadable file.

In architecture terms: **UI → Workflow pod (5 nodes, 3 amber gates)
→ (3 Groq calls + 1 ChromaDB query + 1 ML call) → UI**. Roughly
30–60 seconds end-to-end on free-tier Groq.

---

#### Flow E — User clicks "Generate & Download PDF Report"

Shortest non-trivial flow on the diagram. Note that **pdf_generator
itself does not talk to the ML layer** — the UI handler is the one
making the prediction, exactly the same way Tab 1 does.

1. The UI follows the same *"Single Patient / Batch tab"* dashed
   arrow (via the `predict_noshow` tool) to the **ML layer** to get
   the patient's probability and tier.
2. The UI grabs the most recent assistant message from the chat as
   the plan body (defaulting to a placeholder if the chat is empty).
3. The UI then follows the *"Generate & Download PDF"* arrow to
   **pdf_generator**, handing it the pre-computed risk numbers and
   the plan text.
4. pdf_generator assembles the PDF — patient summary, risk box, plan
   text, and the ethical disclaimer — and hands the file back.
5. The finished PDF travels back to the UI and the browser downloads it.

The PDF button **does not** re-run the full Workflow pod — it reuses
whichever chat message or plan is already on screen. For a fully
grounded, citation-backed PDF, the user should run Flow D first and
then click the PDF button.

---

#### Flow F — Offline: building the RAG index (before the app ever runs)

This is the only flow that doesn't start from a user click. Trace the
three arrows inside the **RAG Pipeline pod** from left to right:

1. A maintainer runs the vector-store builder.
2. The builder reads the five Markdown files under *guidelines/*.
3. The *"chunk"* arrow sends them through the **RecursiveCharacterText
   Splitter** (chunk 500, overlap 80), producing **40 chunks**.
4. The *"embed"* arrow sends the chunks through **all-MiniLM-L6-v2**,
   turning each chunk into a 384-dimensional vector.
5. The *"persist"* arrow writes the vectors into **ChromaDB**.
6. The populated ChromaDB folder is committed to the Git repository,
   so every subsequent deploy ships with a ready-to-use knowledge base.

After this one-time run, Flows C and D can hit a populated ChromaDB
instantly — no rebuilding required.

---

### Q. What is a "threshold"?

A threshold is just a **cut-off number** — a line that separates one
thing from another.

- Everyday example: *"anyone above 18 can vote."* The number 18 is a
  threshold. Below 18 you can't vote; above 18 you can.
- In our project: the Decision Tree gives every patient a number
  between 0 and 1 that says *"how likely is this patient to miss the
  appointment?"*
  - 0.05 = very unlikely
  - 0.47 = unclear
  - 0.87 = very likely

But a number alone doesn't tell the front-desk staff what to **do**,
so we use **two cut-off lines** to turn it into three buckets with
three different actions.

| Model's number | Bucket (tier) | What the clinic does |
|---|---|---|
| Less than **0.30** | LOW    | Send the usual automated SMS only |
| Between **0.30** and **0.55** | MEDIUM | Personal SMS + maybe a courtesy call |
| More than **0.55** | HIGH   | Call the patient to confirm |

**0.30** and **0.55** are the two thresholds. They're just cut-offs.
We didn't invent them — they come straight from the outreach protocol
in `attendance_management.md`.

### Q. If the Decision Tree already gives a risk number, why do we need the `risk_tier` gate?

The Decision Tree gives us a **continuous probability** — a number
like 0.47. That number is **evidence**. It is not a decision.

The `risk_tier` gate takes the probability and applies the two
thresholds above to produce a **decision** — LOW, MEDIUM, or HIGH —
which the downstream retrieval step then uses to pull the right
policy.

**Tree outputs evidence. Gate outputs a decision.** Two different jobs.

### Q. If ChromaDB is called only once, why do we have **two** gates after it?

Because the two gates guard **two different things**:

- The **retrieval-empty gate** (`HALL`, first amber diamond) checks
  the *input* to the LLM — did the vector store give us any evidence
  to ground the plan in? If the retriever returns zero documents, the
  agent has nothing to cite and must refuse.
- The **Source-tag gate** (`CIT`, second amber diamond) checks the
  *output* of the LLM — after the LLM wrote the plan, did it actually
  include `[Source: filename]` citations for every policy claim?

So the two gates protect against two different failure modes:

| Gate | Failure it prevents |
|---|---|
| `HALL` (retrieval empty) | The LLM is given no evidence at all and hallucinates policy from training data. |
| `CIT` (Source tag missing) | The LLM is given evidence but ignores it and writes uncited claims anyway. |

Both can happen independently — retrieval could return five excellent
chunks and the LLM could still produce an uncited plan. In the
current implementation, `CIT` is enforced by the hardened system
prompt (the LLM is told explicitly to cite every claim). The dashed
`no` arrow looping back to `intervention_plan` represents a
"re-ask until cited" guardrail that we plan to harden further.

### Q. The PDF generator doesn't have an arrow to the ML layer — how does it get the risk number?

It doesn't. **The UI is the one calling the ML layer, not the PDF
generator.** When the user clicks the PDF button, the UI first
follows the same arrow that Tab 1 uses (the
`predict_noshow`-tool / dashed arrow into the ML layer), gets the
probability and tier back, and then hands those numbers plus a
plan-text body to `pdf_generator` along the solid *"Generate &
Download PDF"* arrow.

The PDF generator is a pure formatter — it takes data in and emits a
PDF. It does not reach back into the model by itself.

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

### Q. What is Random Forest, and why didn't you use it?

**Random Forest in one sentence.** It's an **ensemble of many
Decision Trees** — typically between 100 and 1{,}000 trees — whose
votes are averaged to produce the final prediction. Each tree sees a
random sample of the rows *and* a random subset of the features,
which forces the trees to disagree in useful ways. The averaged
prediction is usually more accurate and more stable than any single
tree.

**How it differs from a single Decision Tree.**

| | Single Decision Tree | Random Forest |
|---|---|---|
| Number of trees | 1 | 100+ |
| Variance | High (overfits small data patterns) | Low (averaging cancels the noise) |
| Interpretability | Easy — you can print and read it | Hard — 100 trees is a black box |
| Inference speed | Microseconds | Slower by the number of trees |
| Memory | Tiny (~75 KB) | Bigger by the number of trees |

**Why we stuck with a single Decision Tree for *this* project.**

1. **Interpretability — what this actually means.** After training,
   you can ask a Decision Tree *"which features mattered most for
   your predictions?"* and it hands you back a clean ranked list:
   ```
   1. AwaitingTime     — 35 %
   2. Num_App_Missed   — 30 %
   3. Age              — 10 %
   4. SMS_received     —  8 %
   5. ...
   ```
   One tree, one clear list — easy to put in the report, easy to
   defend to a professor. A Random Forest is 100+ trees, each with
   its own ranking; to explain the forest you have to average all
   100 rankings together. That average is still useful, but no single
   tree inside the forest can walk you through its reasoning.
   **Simple analogy:** one tree is one doctor's opinion — you can ask
   them to explain it. A forest is 100 doctors voting — you know
   what the group thinks, but no single doctor can show you the logic.
2. **Inference budget** — the model is invoked as a *tool* by the
   LLM agent on every turn. A single tree takes microseconds; 100
   trees take visibly longer.
3. **Rubric focus** — the end-sem rubric rewards the agentic layer
   (35 %), not raw ML performance. Spending complexity on a forest
   would not change the grade much; spending it on the agent does.
4. **Honest future-work note** — a class-weighted Random Forest would
   probably lift minority-class recall by a few points. If a grader
   pushes, just say: *"One-evening follow-up experiment; we
   prioritised the agentic layer instead."*

### Q. Why `max_depth=10`?

Shallow enough to avoid overfitting 100K+ rows, deep enough to capture feature interactions like *young + long lead + prior miss*. Empirically, test accuracy plateaus around depth 10 and falls past depth 15 due to overfitting. It's also what the original exploration notebook converged on.

### Q. What is balanced vs unbalanced data? (Plain English)

**Balanced data** = each category has roughly the same number of
examples.
> *Example:* a classroom of 50 boys and 50 girls is balanced.

**Unbalanced data** = one category is much bigger than the other.
> *Example:* a classroom of 80 boys and 20 girls is unbalanced.

**Our dataset is unbalanced.** Out of every 100 patients:

- **80 show up** to their appointment.
- **20 miss** their appointment.

So *"show up"* is the big group, *"no-show"* is the small group.

### Q. Why is unbalanced data a problem?

Because an un-careful model learns a cheap trick:

> *"I'll predict 'show up' for every patient. I'll be right 80 % of
> the time — 80 % accuracy!"*

On paper that model looks great. In real life it is **useless** —
it never correctly identifies any no-shows, and catching no-shows is
the whole point. The model has to work harder than *"always say the
big group"*.

### Q. What are the three standard fixes for unbalanced data?

**Fix 1 — Oversampling.**
Make the small group bigger by **copying its rows**. Take the 20
no-show rows in our data and duplicate them until there are 80. Now
both groups have 80 examples each and the model sees both equally.
A smarter version called **SMOTE** creates slightly modified
synthetic copies instead of exact duplicates, but the idea is the
same: boost the small group by adding more examples of it.

**Fix 2 — `class_weight='balanced'`.**
Tell the model *"pay extra attention to the small group."*

> Think of a teacher grading an exam. Normally each question is
> worth 1 point. For the really important topics the teacher says
> *"this question is worth 4 points."* The student pays extra
> attention to those. `class_weight='balanced'` does the same —
> when the model makes a mistake on a no-show it is penalised 4×
> harder, so it learns to not miss them.

**Fix 3 — Threshold calibration.** *(This is what we used.)*
Remember the Decision Tree gives a number between 0 and 1. The
**default** rule in scikit-learn is *"if that number is above 0.50,
predict no-show; below, predict show-up."* So 0.50 is the default
threshold.

Threshold calibration means **picking a different threshold** that
works better for our situation. Instead of one cut-off at 0.50, we
use **two** cut-offs — 0.30 and 0.55 — which gives us three action
levels (LOW, MEDIUM, HIGH) instead of a yes/no.

### Q. Why did we pick threshold calibration (Fix 3) over the other two?

Because the clinic doesn't want a yes/no answer. They want **three
different actions** depending on how worried they should be:

- LOW worry → just send an SMS.
- MEDIUM worry → SMS + maybe a courtesy call.
- HIGH worry → pick up the phone.

A single yes/no classifier (which is what `class_weight='balanced'`
or oversampling would give us) can't produce three tiers. Our two
thresholds can. The thresholds also line up exactly with what
`attendance_management.md` says to do at each risk level.

### Q. Short answer for the viva

> *"Our data is unbalanced — 80 % show up, 20 % don't. We didn't
> fix it with `class_weight='balanced'` or oversampling, because
> those give a yes/no answer. Instead we used threshold calibration
> — we picked two cut-offs, 0.30 and 0.55, that map the model's
> probability to three action levels (LOW, MEDIUM, HIGH) matching
> the clinic's outreach protocol."*

If they push further: *"class_weight is a one-line change and we'd
run it as a follow-up experiment."*

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

### Q. What is the difference between **encoding** and **embedding**?

They both turn something non-numeric into numbers, but they answer
very different questions.

**Encoding — structural conversion.**
A fixed lookup from a category to a number. No learning involved. The
numbers are arbitrary placeholders; two categories with similar
numbers are *not* necessarily similar in meaning.

- *One-hot encoding* — each category becomes a separate binary column.
  e.g. `Gender` → `Gender_F = [1,0]`, `Gender_M = [0,1]`. This is what
  we used for the ML features in our project.
- *Label encoding* — each category is mapped to a single integer
  (A → 0, B → 1, C → 2, …). Cheap, but imposes a false ordering.

**Embedding — learned semantic vector.**
A **dense vector of real numbers** (e.g. 384 numbers per piece of
text) produced by a **trained neural network**. The coordinates are
chosen so that *semantically similar inputs are close together* in
vector space. You cannot read an individual dimension — the meaning
lives in the geometry.

- *Sentence embedding* — what our RAG pipeline uses.
  `all-MiniLM-L6-v2` turns each 500-character chunk of a guideline
  document into a 384-dimensional vector. *"SMS reminder policy for
  medium-risk patients"* and *"how do we remind patients before an
  appointment?"* produce **similar** vectors even though they share
  few words.

**One-line summary.**

| | Encoding | Embedding |
|---|---|---|
| Purpose | Structural (categorical → numeric) | Semantic (meaning → geometry) |
| Learned? | No, it's a fixed lookup | Yes, from a trained model |
| Dimensions | Usually ≤ number of categories | Hundreds (e.g. 384) |
| Similar meanings → similar numbers? | No | Yes |
| Example in our project | `Gender_F`, `Gender_M`, `Handicap_0..4` | ChromaDB's 384-d vectors of guideline chunks |

**Why both appear in our project.**
Encoding prepares the *tabular* patient features for the Decision
Tree. Embedding powers the *semantic* search over the policy
documents. They coexist because they solve different problems.

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
