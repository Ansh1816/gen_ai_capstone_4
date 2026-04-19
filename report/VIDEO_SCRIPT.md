# 5-Minute Demo Video Shooting Script

**Runtime target:** 5:00 (exactly, ± 5 seconds)
**Format:** Screen recording with voiceover
**Tool suggestion:** QuickTime Player (Mac, built-in) or OBS Studio (free), plus any external mic.

---

## Structure at a glance

| Segment | Time | Duration |
|---|---|---|
| 1. Opening title | 0:00 – 0:10 | 10 s |
| 2. Problem statement | 0:10 – 0:40 | 30 s |
| 3. Codebase tour + rubric checklist | 0:40 – 1:20 | 40 s |
| 4. Streamlit demo — Single Patient tab | 1:20 – 2:05 | 45 s |
| 5. Streamlit demo — Batch Analysis tab | 2:05 – 2:25 | 20 s |
| 6. Streamlit demo — AI Care Coordinator chat (RAG) | 2:25 – 3:20 | 55 s |
| 7. Streamlit demo — Run Full Care Workflow (5-step) | 3:20 – 4:40 | 80 s |
| 8. Guardrails + honest failure-mode retelling | 4:40 – 4:55 | 15 s |
| 9. Closing | 4:55 – 5:00 | 5 s |

**Most of the runtime — 3:20 out of 5:00 (66%) — is the live Streamlit demo.**

---

## Pre-record checklist

1. **Rehearse once through** reading the script aloud with a stopwatch. Aim for ~140 words per minute.
2. **Warm up the hosted app** — open https://genaicapstone4-anot5dcxcx9vhuxzsifnzb.streamlit.app/ about 5 minutes before recording so it's not stuck on the hibernation "wake up" screen.
3. **Pre-fill the AI Care Coordinator form** before you hit record so the demo flows without fumbling:
   - Age: 53, Gender: Female, Lead Time: 47, SMS Reminder: ON, Hypertension: ON, Diabetes: ON, Scholarship: ON, Disability Level: 1, Previous No-Shows: 1.
4. **Open three browser tabs**, in this order, so you can Cmd+1 / Cmd+2 / Cmd+3 between them:
   - Tab 1 — Live Streamlit app: https://genaicapstone4-anot5dcxcx9vhuxzsifnzb.streamlit.app/
   - Tab 2 — GitHub repo: https://github.com/Ansh1816/gen_ai_capstone_4
   - Tab 3 — PDF report: open `report/report.pdf` in Preview.
5. **Close every other window.** Disable notifications (macOS → Focus → Do Not Disturb).
6. **Record at 1080p minimum.** QuickTime: File → New Screen Recording → click the tiny arrow next to the red record button → pick your external mic → Start Recording.

---

## 0:00 – 0:10 · Opening title (10 s)

**On screen:** Live Streamlit app (Tab 1), parked on the dark-blue header — nothing clicked yet.

> *"Hi, I'm Praanshu Ranjan. This is a 5-minute walkthrough of our end-semester capstone: an intelligent appointment no-show prediction and agentic care coordination assistant."*

---

## 0:10 – 0:40 · Problem statement (30 s)

**On screen:** Stay on the live app header for the first 10 seconds, then switch to **Tab 2 (GitHub)** to set up the transition for section 3.

> *"The problem first. About one in every five patients misses their scheduled medical appointment. That's a huge number. For a hospital, missed appointments mean lost revenue, wasted clinician time, and a slot that could have gone to another patient in need. For the patient, it means a break in their care — chronic conditions go unmonitored, prescriptions run out, and small issues become emergencies.*
> *Our system does two things. One, it predicts which patients are most likely to miss an upcoming appointment using their booking history and demographics. Two — and this is the agentic part — it tells the front-desk staff exactly what to do about each at-risk patient, using the hospital's own written rule-book as the source of truth, not the AI's imagination."*

---

## 0:40 – 1:20 · Codebase tour + rubric checklist (40 s)

**On screen:** GitHub repo, scrolled so the top of the file list is visible. Slowly scroll down as you mention each item, hovering the cursor on the file you're naming.

> *"Before I show the app, let me quickly walk through the repo and check off the end-semester rubric.*
> *Rubric item one — the machine-learning prediction module. `model_brain.py` trains the Decision Tree; `model.pkl`, `scaler.pkl`, and `feature_cols.pkl` are the persisted artefacts.*
> *Two — LangGraph agentic workflow with explicit state. That's `agent.py` — a five-node StateGraph with a strongly-typed AgentState TypedDict.*
> *Three — ReAct agent with tool-use. That's `chatbot_agent.py` with the `predict_noshow` and `search_guidelines` tools.*
> *Four — Retrieval-Augmented Generation using Chroma. The corpus is these five markdown files under `guidelines/` — attendance, reminders, overbooking, engagement, and ethics. The pre-built vector store is committed under `chroma_db/` so it ships with the deployment.*
> *Five — structured output with risk summary, intervention plan, sources, and ethical disclaimer. That's the Step Five node of the workflow plus `pdf_generator.py`.*
> *Six — documentation. The README on the front page has the live-app badge, architecture diagram, setup, and team credits. The 23-page LaTeX report is in the `report` folder. Every file the rubric asks for is here."*

---

## 1:20 – 2:05 · Streamlit demo — Single Patient Analysis tab (45 s)

**On screen:** Switch back to **Tab 1 (live app)**. Click the **SINGLE PATIENT ANALYSIS** tab. Enter a patient — Age 65, Female, Lead Time 30, toggle Hypertension and Diabetes to ON. Click **Analyse Appointment Risk**.

> *"Now the app. Tab one is the classical machine-learning layer. I fill in the patient on the left — age, gender, lead time, SMS reminder, chronic conditions, and prior no-show history. The Decision Tree model takes fourteen features in total, including two we engineered ourselves: `AwaitingTime`, the number of days between booking and appointment, and `Num_App_Missed`, the cumulative count of past no-shows for this patient.*
> *I click Analyse. On the right, the probability, the risk tier — Low, Medium, or High — and a rule-based primary action. The scenario simulator below lets me slide the lead time or the prior-miss count and watch the risk change in real time — useful when front-desk staff are negotiating an earlier slot with the patient."*

[Demonstrate sliding **Lead Time** in the scenario panel from 30 down to 5 — pause for 1 second so the grader sees the delta badge update]

---

## 2:05 – 2:25 · Streamlit demo — Batch Analysis tab (20 s)

**On screen:** Click the **BATCH ANALYSIS** tab.

> *"Tab two is for hospitals that want to run the model over a full day's schedule at once. Upload a CSV, and you get KPI cards for total appointments, high-risk count, estimated no-shows, and suggested standby slots — plus a sortable high-risk patient list you can export. Same Decision Tree, just vectorised across every row."*

---

## 2:25 – 3:20 · Streamlit demo — AI Care Coordinator chat with RAG (55 s)

**On screen:** Click the **AI CARE COORDINATOR** tab. Scroll to the chat input at the bottom of the right panel.

> *"Tab three is where this stops being a mid-sem project and becomes an end-sem one. This is the agentic layer. I can chat freely with the Care Coordinator — it has the machine-learning model and the hospital rule-book as tools."*

[Type exactly]: **What is the reminder policy for high-risk patients? Cite the source document.**

[Press enter. Let the spinner run.]

> *"Under the hood this is a LangGraph ReAct loop. The agent sees my question, decides which tool to call, and because I'm asking about a policy, it calls `search_guidelines`. That tool runs a similarity search over the ChromaDB vector store — forty embedded chunks — and returns the matching excerpts with their source filenames attached.*
> *Here comes the answer. Notice two things. One, the specific protocol is there — twenty-four-hour pre-appointment call, four-hour confirmation deadline, overbooking authorization. Two, every claim carries the source file in brackets — `attendance_management.md` and `reminder_policy.md`. Nothing here is made up."*

[Click the **View Agent Logic & Tool Usage** expander for 2 seconds to reveal the tool call trace, then close it.]

---

## 3:20 – 4:40 · Streamlit demo — Run Full Care Workflow (80 s)

**On screen:** On the left panel, scroll down to the ACTION OPTIONS card. The patient should already be set to Age 53, Female, 47-day lead time, hypertension + diabetes + scholarship, Disability 1, Prior No-Shows 1. Click **Run Full Care Workflow**.

> *"The chat is the free-form surface. The real showpiece is this button — Run Full Care Workflow. It executes a five-step LangGraph StateGraph with a typed AgentState that flows through every node. Let me run it on a harder patient — fifty-three-year-old female, forty-seven-day lead time, hypertension and diabetes, on scholarship, one previous no-show."*

[Wait for the workflow — usually 30 to 60 seconds. Spinner visible.]

> *"Five steps in sequence. Step One — risk assessment — the Decision Tree runs and produces the probability and tier.*
> *Step Two — an LLM call that explains why this patient is at risk, anchored to their actual values. You can see it references the forty-seven-day lead time, the prior miss, and the scholarship flag specifically, not generic advice.*
> *Step Three is the agentic centrepiece. The agent builds a query from the tier and the factors, retrieves the top five chunks from Chroma, and displays them here as source chips — you can see all five filenames — plus the exact excerpts that were retrieved.*
> *Step Four takes those excerpts plus the Step-Two analysis and produces a structured intervention plan — primary action, numbered strategies, timeline, escalation protocol, expected outcome. Every policy claim in the plan is tagged with a `Source` filename.*
> *Step Five compiles the full report ready for export."*

[Scroll down so the permanent Operational & Ethical Notice card is visible.]

> *"And beneath the five steps, a permanent ethical disclaimer: AI decision support only, every recommendation reviewed by qualified staff, no medical advice."*

---

## 4:40 – 4:55 · Guardrails + honest deployment-hazard story (15 s)

**On screen:** Stay on the AI Care Coordinator tab.

> *"One honest note. When we first deployed, the ChromaDB folder was excluded from Git, so the live agent had an empty vector store and was confidently making up fake hospital policies. I diagnosed that, committed the populated vector store, and hardened the system prompt to refuse to invent policies. The full story is in Section 9 of the project report."*

---

## 4:55 – 5:00 · Closing (5 s)

**On screen:** Briefly show Tab 2 (GitHub) one last time.

> *"Code, live app, report, and this video are all linked from the repository. Thanks."*

---

## Post-record checklist

1. **Trim** the recording to exactly 5 minutes — remove any dead air at the very start and end.
2. **Normalise audio** — Audacity or DaVinci Resolve (both free) to level out the volume.
3. **Export as MP4 H.264**, 1080p, ~30 fps. Target file size under 200 MB.
4. **Upload to:**
   - **YouTube (unlisted)** — easiest for sharing a link, OR
   - **Google Drive** with link-sharing on, OR
   - **Vimeo** free tier.
5. **Add a badge to the README** once you have the URL:
   ```markdown
   [![Demo Video](https://img.shields.io/badge/Demo-5%20min-red?logo=youtube)](YOUR_VIDEO_URL)
   ```

---

## Pacing tips

- Script is **≈780 words**. At ~155 wpm that's exactly 5:00.
- **Breathe between sentences.** It feels slow to you, but a grader hears it as confident.
- **Slow cursor movement.** The grader is trying to read the screen while you talk. Pause 1–2 seconds on anything they need to read.
- If any section runs long in rehearsal, **cut Batch Analysis first** (20 s drop) — it's the least distinctive tab.
- If the codebase section runs short (because you speak quickly), **hover longer** on each file name as you say it. The grader is building a mental map of the repo from this segment.

---

## If things go wrong during recording

- **Spinner stalls on Run Full Care Workflow:** browser refresh, redo the patient, retry. The free-tier Groq API occasionally throttles.
- **App is hibernating:** the "Yes, get this app back up!" screen appears — click it, wait 30–60 seconds, then restart the recording from the beginning.
- **You stutter a word:** pause for 2 seconds and say the sentence again. The visible gap in the audio waveform makes the bad take easy to cut out in editing.
