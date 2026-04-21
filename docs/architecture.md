# System Architecture

End-to-end view of the **Intelligent Appointment No-Show Prediction & Agentic
Care Coordination Assistant**. Every node below is clickable — it links to
the file that implements it.

```mermaid
flowchart TD
    %% Native GitHub theme adaptation
    classDef default fill:transparent,stroke:#52525B,stroke-width:1px,rx:6px,ry:6px;
    classDef ui fill:transparent,stroke:#3b82f6,stroke-width:2px,rx:6px,ry:6px;
    classDef ext fill:transparent,stroke:#8b5cf6,stroke-width:2px,stroke-dasharray: 4 4,rx:6px,ry:6px;
    classDef agent fill:transparent,stroke:#14b8a6,stroke-width:2px,rx:6px,ry:6px;
    classDef util fill:transparent,stroke:#64748b,stroke-width:2px,rx:6px,ry:6px;
    classDef gate fill:transparent,stroke:#f59e0b,stroke-width:2px,rx:6px,ry:6px;

    %% UI routing
    UI[Streamlit UI<br/>single · batch · coordinator]:::ui -->|Single Patient / Batch tab| ML[DecisionTreeClassifier<br/>model.pkl]:::util
    UI -->|Run Full Care Workflow| RA
    UI -->|Chat message| REACT[ReAct Agent<br/>chatbot_agent.py]:::agent
    UI -->|Generate & Download PDF| PDF

    %% 5-Step LangGraph Workflow pod
    subgraph Workflow [5-Step Care Workflow · agent.py]
        RA[risk_assessment]:::util --> TIER{risk_tier<br/>0.30 / 0.55}:::gate
        TIER -->|LOW · MEDIUM · HIGH| RR[risk_reasoning]:::agent
        RR --> GR[guideline_retrieval]:::util
        GR --> HALL{retrieval empty?}:::gate
        HALL -->|docs found| IP[intervention_plan]:::agent
        HALL -.->|zero docs| REFUSE[inform user<br/>no invented policy]:::gate
        IP --> CIT{carries<br/>Source tag?}:::gate
        CIT -->|yes| RG[report_generation]:::agent
        CIT -.->|no| IP
        RG --> FIN([structured output<br/>+ disclaimer]):::util
    end
    style Workflow fill:none,stroke:#52525B,stroke-width:1px,stroke-dasharray:5 5,color:#A1A1AA

    %% RAG Pipeline pod (offline + runtime)
    subgraph RAG [RAG Pipeline · build_vectorstore.py]
        DOCS[(guidelines/*.md<br/>5 policy documents)]:::util --> CHUNK[RecursiveCharacterTextSplitter<br/>chunk=500 · overlap=80]:::util
        CHUNK --> EMBED[all-MiniLM-L6-v2<br/>384-d embeddings]:::util
        EMBED --> CHROMA[(ChromaDB<br/>40 chunks · persisted in-repo)]:::ext
    end
    style RAG fill:none,stroke:#52525B,stroke-width:1px,stroke-dasharray:5 5,color:#A1A1AA

    %% Cross-pod retrieval
    GR -.->|top-k=5 similarity| CHROMA
    REACT -.->|search_guidelines tool| CHROMA
    REACT -.->|predict_noshow tool| ML

    %% LLM backend
    GROQ[(Groq Cloud<br/>llama-3.1-8b-instant<br/>free tier)]:::ext
    RR -.->|temp 0.3| GROQ
    IP -.->|temp 0.3| GROQ
    RG -.->|temp 0.3| GROQ
    REACT -.->|reasoning loop| GROQ

    %% Output
    FIN --> PDF[pdf_generator.py<br/>fpdf2]:::util

    %% Interactivity — click nodes to jump into the repo
    click UI "../app.py" "Streamlit UI entry point"
    click ML "../model_brain.py" "DecisionTreeClassifier training + inference"
    click RA "../agent.py" "Node 1 — feature build + predict_proba"
    click TIER "../agent.py" "Thresholds 0.30 and 0.55 set the tier"
    click RR "../agent.py" "Node 2 — LLM risk-factor analysis"
    click GR "../agent.py" "Node 3 — top-5 chunks from Chroma"
    click HALL "../chatbot_agent.py" "Empty-retrieval fallback string"
    click IP "../agent.py" "Node 4 — grounded intervention plan"
    click CIT "../chatbot_agent.py" "Citation guardrail — every claim tagged"
    click RG "../agent.py" "Node 5 — final structured report + disclaimer"
    click REFUSE "../chatbot_agent.py" "System prompt blocks invented policies"
    click REACT "../chatbot_agent.py" "ReAct agent with predict_noshow + search_guidelines"
    click DOCS "../guidelines/" "5 hospital-operations markdown files"
    click CHUNK "../build_vectorstore.py" "Markdown-aware splitter"
    click EMBED "../build_vectorstore.py" "HuggingFace sentence-transformers on CPU"
    click CHROMA "../chroma_db/" "Committed so the hosted app ships populated"
    click PDF "../pdf_generator.py" "Downloadable audit report"
    click GROQ "https://groq.com" "Free-tier Llama 3.1 8B inference"
```

---

## Legend

| Colour | Role | Examples |
|---|---|---|
| **Blue** | User-facing surface | Streamlit UI |
| **Teal** | LLM-driven agent node | `risk_reasoning`, `intervention_plan`, `report_generation`, ReAct agent |
| **Slate** | Deterministic processing | `risk_assessment` (ML), `guideline_retrieval`, splitter, embedder, PDF generator |
| **Amber** | Decision / guardrail | risk-tier threshold, empty-retrieval check, citation enforcement |
| **Purple (dashed)** | External service | Groq Cloud, ChromaDB |

**Arrow semantics**

- **Solid** `-->` — primary control flow (happy path).
- **Dashed** `-.->` — background I/O, tool calls, guardrail fallbacks, and re-ask loops.

---

## Layer walk-through

### Streamlit UI (`app.py`)

Three tabs. The first two go straight to the ML model; the third surfaces
both the ReAct chat agent and a **Run Full Care Workflow** button that
invokes the full 5-step LangGraph pipeline.

### 5-Step Care Workflow (`agent.py`)

A LangGraph `StateGraph` with an explicit `AgentState` `TypedDict`:

1. **`risk_assessment`** — Decision Tree predicts no-show probability, thresholds
   pick a tier.
2. **`risk_reasoning`** — LLM explains the prediction in plain English anchored
   to the patient's actual values.
3. **`guideline_retrieval`** — builds a tier- and factor-aware query, pulls the
   top-5 chunks from ChromaDB.
4. **`intervention_plan`** — LLM grounds the recommendation in the retrieved
   excerpts; every claim carries a `[Source: filename]` tag.
5. **`report_generation`** — structured final report with patient summary,
   risk, reasoning, plan, sources, and ethical disclaimer.

### Conversational ReAct Agent (`chatbot_agent.py`)

Built with `langgraph.prebuilt.create_react_agent`. Two tools:

- **`predict_noshow`** — calls the Decision Tree; returns `{risk_score, risk_tier}`.
- **`search_guidelines`** — similarity search over ChromaDB; returns
  `[Source: <filename>]`-prefixed excerpts, or the literal fallback
  `"No specific guidelines found for this query."` when empty.

The system prompt hardens the behaviour: require exact probabilities, require
source tags on every policy claim, refuse to invent policy when retrieval is
empty, and always append the operational-and-ethical disclaimer.

### RAG Pipeline (`build_vectorstore.py`)

Offline, one-shot. Chunks the five markdown documents under `guidelines/`
(chunk 500, overlap 80), embeds with `sentence-transformers/all-MiniLM-L6-v2`
on CPU, and persists into `chroma_db/`. The populated store is committed
in-repo so the hosted container ships with a ready-to-use knowledge base on
first boot.

### Guardrails

Five enforcement points, four of which are drawn as amber gates above:

| Guardrail | Where | Purpose |
|---|---|---|
| Risk tier | `risk_assessment` output | Maps probability to operational tier. |
| Retrieval empty check | `search_guidelines` tool | Emits a literal fallback rather than silence. |
| Citation enforcement | ReAct system prompt | Every policy claim must carry `[Source: filename]`. |
| Hallucination refusal | ReAct system prompt | Must say "no matching policy" rather than invent. |
| Disclaimer | every structured output | Operational + ethical notice on every report. |

---

## Alternate renders

If your viewer does not support Mermaid, the same architecture is available as
a high-resolution PNG and SVG in the `report/` directory:

- [`report/fig_architecture_graphviz.png`](../report/fig_architecture_graphviz.png) — 200-DPI PNG
- [`report/fig_architecture_graphviz.svg`](../report/fig_architecture_graphviz.svg) — scalable vector
- [`report/fig_architecture.dot`](../report/fig_architecture.dot) — Graphviz source
- [`report/fig_architecture.mmd`](../report/fig_architecture.mmd) — plain Mermaid source
