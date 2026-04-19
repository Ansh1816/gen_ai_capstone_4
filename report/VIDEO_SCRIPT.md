# 5-Minute Demo Video Shooting Script

**Runtime target:** 5:00 (exactly, ± 5 seconds)
**Format:** Screen recording with voiceover
**Voice:** Team voice throughout — use "we" and "our". No individual names, no personal attribution.
**Tool suggestion:** QuickTime Player (Mac, built-in) or OBS Studio (free), plus any external mic.

---

## Structure at a glance

| Segment | Time | Duration |
|---|---|---|
| 1. Opening title | 0:00 – 0:10 | 10 s |
| 2. Problem statement | 0:10 – 0:40 | 30 s |
| 3. Codebase tour + rubric checklist | 0:40 – 1:15 | 35 s |
| 4. Streamlit demo — Tab 1: Single Patient Analysis | 1:15 – 2:00 | 45 s |
| 5. Streamlit demo — Tab 2: Batch Analysis | 2:00 – 2:45 | 45 s |
| 6. Streamlit demo — Tab 3a: AI Care Coordinator chat (RAG) | 2:45 – 3:35 | 50 s |
| 7. Streamlit demo — Tab 3b: Run Full Care Workflow (5-step) | 3:35 – 4:50 | 75 s |
| 8. Guardrails note | 4:50 – 4:55 | 5 s |
| 9. Closing | 4:55 – 5:00 | 5 s |

**Streamlit demo = 3:35 out of 5:00 (72%), evenly split across the three tabs:**
**45 s · 45 s · 125 s (50 s chat + 75 s workflow).**

---

## Pre-record checklist

1. **Rehearse once through** with a stopwatch. Aim for ~150 words per minute.
2. **Warm up the hosted app** — open the live Streamlit URL about 5 minutes before recording so it's not on the hibernation screen.
3. **Prepare a sample CSV for the Batch Analysis demo.** Download `KaggleV2-May-2016.csv` from Kaggle (or copy from our local training data) and create a small 20-row slice called `batch_demo.csv` — the tab expects KaggleV2 schema columns including `PatientId, Age, Gender, Hipertension, Diabetes, Scholarship, Handcap, SMS_received, ScheduledDay, AppointmentDay, No-show`. Have the file on the desktop so it's one drag away during the recording.
4. **Pre-fill the AI Care Coordinator form** before rolling:
    - Age: 53, Gender: Female, Lead Time: 47, SMS Reminder: ON, Hypertension: ON, Diabetes: ON, Scholarship: ON, Disability Level: 1, Previous No-Shows: 1.
5. **Open three browser tabs** (Cmd+1 / Cmd+2 / Cmd+3):
    - Tab 1 — Live Streamlit app
    - Tab 2 — GitHub repo
    - Tab 3 — PDF report (`report/report.pdf` in Preview)
6. **Close every other window.** macOS → Focus → Do Not Disturb.
7. **Record at 1080p minimum.** QuickTime → File → New Screen Recording → pick external mic → Start.

---

## 0:00 – 0:10 · Opening title (10 s)

**On screen:** Live Streamlit app (Tab 1), parked on the dark-blue header — nothing clicked yet.

> *"This is a 5-minute walkthrough of our end-semester capstone: an intelligent appointment no-show prediction and agentic care coordination assistant."*

---

## 0:10 – 0:40 · Problem statement (30 s)

**On screen:** Stay on the live app header for ~10 seconds, then switch to **Tab 2 (GitHub)** to set up the next section.

> *"The problem first. About one in every five patients misses a scheduled medical appointment. That's a huge number. For a hospital, missed appointments mean lost revenue, wasted clinician time, and a slot that could have gone to another patient in need. For the patient, it means a break in their care — chronic conditions go unmonitored, prescriptions run out, and small issues become emergencies.*
> *Our system does two things. First, it predicts which patients are most likely to miss an upcoming appointment using their booking history and demographics. Second — and this is the agentic part — it tells the front-desk staff exactly what to do about each at-risk patient, using the hospital's own written rule-book as the source of truth, not the AI's imagination."*

---

## 0:40 – 1:15 · Codebase tour + rubric checklist (35 s)

**On screen:** GitHub repo, scrolled so the top of the file list is visible. Slowly scroll down, hovering the cursor on the file you're naming.

> *"Before the live app, a quick walk-through of the repository, ticking off the end-semester rubric as we go.*
> *Rubric item one — the machine-learning prediction module. `model_brain.py` trains the Decision Tree; `model.pkl`, `scaler.pkl`, and `feature_cols.pkl` are the persisted artefacts.*
> *Two — LangGraph agentic workflow with explicit state. That's `agent.py` — a five-node StateGraph with a strongly-typed AgentState TypedDict.*
> *Three — ReAct agent with tool-use. That's `chatbot_agent.py` with the `predict_noshow` and `search_guidelines` tools.*
> *Four — Retrieval-Augmented Generation with Chroma. The corpus is these five markdown files under `guidelines/`. The pre-built vector store is committed under `chroma_db/` so it ships with the deployment.*
> *Five — structured output with risk summary, intervention plan, sources, and ethical disclaimer. That's the Step Five node plus `pdf_generator.py`.*
> *Six — documentation. The README has the live-app badge, architecture diagram, and setup. The 23-page LaTeX report is in the `report` folder. Every file the rubric asks for is here."*

---

## 1:15 – 2:00 · Tab 1 — Single Patient Analysis (45 s)

**On screen:** Switch back to **Tab 1 (live app)**. Click **SINGLE PATIENT ANALYSIS**. Enter Age 65, Female, Lead Time 30, toggle Hypertension and Diabetes ON. Click **Analyse Appointment Risk**.

> *"Tab one is the classical machine-learning layer. A patient's profile goes in on the left — age, gender, lead time, SMS reminder, chronic conditions, and prior no-show history. The Decision Tree model takes fourteen features in total, including two engineered signals: `AwaitingTime`, the days between booking and appointment, and `Num_App_Missed`, the cumulative count of past no-shows.*
> *Clicking Analyse gives us the probability, the risk tier — Low, Medium, or High — and a rule-based primary action. The scenario simulator below lets us slide the lead time or the prior-miss count and watch the risk change in real time — useful when front-desk staff are negotiating an earlier slot with the patient."*

[Demonstrate sliding **Lead Time** from 30 down to 5. Pause 1 second so the delta badge updates.]

---

## 2:00 – 2:45 · Tab 2 — Batch Analysis (45 s)

**On screen:** Click **BATCH ANALYSIS**.

> *"Tab two is built for operational use — running the model over a full day's schedule at once. The clinic exports a CSV from their Hospital Management System and drags it into this upload area."*

[Drag `batch_demo.csv` onto the upload area. Wait ~2 seconds for results to render.]

> *"Once uploaded, the same Decision Tree runs over every row. The results appear as four KPI cards — total appointments, high-risk count, estimated no-shows, and suggested standby slots. Beside them, a Plotly bar chart showing the risk-tier distribution.*
> *Below that, an Operational Actions panel maps each tier to a recommended outreach action with exact headcounts, and a sortable High-Risk Patient List data-frame lets the staff prioritise the day. Finally, a Full Risk Report CSV is exportable for downstream scheduling systems or compliance records. Same machine-learning model as tab one, just vectorised across every appointment in the file."*

[Scroll through the KPI row → chart → actions panel → high-risk list → export button at a steady pace so each card is visible for 2–3 seconds.]

---

## 2:45 – 3:35 · Tab 3a — AI Care Coordinator chat with RAG (50 s)

**On screen:** Click **AI CARE COORDINATOR**. Scroll to the chat input.

> *"Tab three is where the project stops being a mid-sem project and becomes an end-sem one. This is the agentic layer. The Care Coordinator has the machine-learning model and the hospital rule-book as tools."*

[Type exactly]: **What is the reminder policy for high-risk patients? Cite the source document.**

[Press enter. Let the spinner run.]

> *"Under the hood this is a LangGraph ReAct loop. The agent sees the question, decides which tool to call, and because it's a policy question, it calls `search_guidelines`. That tool runs a similarity search over the ChromaDB vector store — forty embedded chunks — and returns matching excerpts with their source filenames attached.*
> *Here's the answer. Two things to notice. The specific protocol is there — twenty-four-hour pre-appointment call, four-hour confirmation deadline, overbooking authorization. And every claim carries the source file in brackets — `attendance_management.md` and `reminder_policy.md`. Nothing here is made up."*

[Click **View Agent Logic & Tool Usage** for 2 seconds, then close it.]

---

## 3:35 – 4:50 · Tab 3b — Run Full Care Workflow (75 s)

**On screen:** Left panel ACTION OPTIONS card. Patient already set (Age 53, Female, 47-day lead time, hypertension + diabetes + scholarship, Disability 1, Prior No-Shows 1). Click **Run Full Care Workflow**.

> *"The chat is the free-form surface. The real showpiece is this button — Run Full Care Workflow. It executes a five-step LangGraph StateGraph with a typed AgentState that flows through every node. Here it runs on a harder patient — fifty-three-year-old female, forty-seven-day lead time, hypertension and diabetes, on scholarship, one previous no-show."*

[Wait for the workflow — 30 to 60 seconds.]

> *"Five steps in sequence. Step One — risk assessment — the Decision Tree runs and produces the probability and tier.*
> *Step Two — an LLM call that explains why this patient is at risk, anchored to their actual values. It references the forty-seven-day lead time, the prior miss, and the scholarship flag specifically, not generic advice.*
> *Step Three is the agentic centrepiece. The agent builds a query from the tier and the factors, retrieves the top five chunks from Chroma, and displays them here as source chips — five filenames — plus the exact excerpts.*
> *Step Four takes those excerpts plus the Step-Two analysis and produces a structured intervention plan — primary action, numbered strategies, timeline, escalation protocol, expected outcome. Every policy claim is tagged with a `Source` filename.*
> *Step Five compiles the full report ready for export."*

[Scroll to show the permanent Operational & Ethical Notice card.]

> *"Beneath the five steps, a permanent ethical disclaimer: AI decision support only, every recommendation reviewed by qualified staff, no medical advice."*

---

## 4:50 – 4:55 · Guardrails note (5 s)

**On screen:** Stay on the AI Care Coordinator tab.

> *"The full failure-mode story and the prompt-hardening fix are documented in Section 9 of the project report."*

---

## 4:55 – 5:00 · Closing (5 s)

**On screen:** Briefly show Tab 2 (GitHub) one last time.

> *"Code, live app, report, and this video are all linked from the repository. Thanks for watching."*

---

## Post-record checklist

1. **Trim** to exactly 5 minutes — remove any dead air at start/end.
2. **Normalise audio** — Audacity or DaVinci Resolve (both free).
3. **Export as MP4 H.264**, 1080p, ~30 fps. Target < 200 MB.
4. **Upload to** YouTube (unlisted) or Google Drive (shared link) or Vimeo.
5. **Add a badge to the README** once you have the URL:
   ```markdown
   [![Demo Video](https://img.shields.io/badge/Demo-5%20min-red?logo=youtube)](YOUR_VIDEO_URL)
   ```

---

## Pacing tips

- Script is ~860 words. At ~170 wpm it lands exactly at 5:00. If that feels fast, drop the guardrails note and close at 4:55.
- **Breathe between sentences.** Feels slow to the speaker, sounds confident to a grader.
- **Slow cursor movement.** Pause 1–2 seconds on anything a grader needs to read.
- If any section runs long in rehearsal, shorten the codebase checklist first by naming only 4 files instead of 6.

---

## If things go wrong

- **Spinner stalls:** refresh, redo patient, retry. Groq free tier occasionally throttles.
- **App is hibernating:** click "Yes, get this app back up!", wait 30–60 s, restart that segment.
- **Batch CSV rejected:** the tab requires a `No-show` column. Easiest fix — copy the training CSV header row into the demo file.
- **Stutter:** pause 2 seconds and repeat the sentence. The audio gap makes cutting the bad take easy.
