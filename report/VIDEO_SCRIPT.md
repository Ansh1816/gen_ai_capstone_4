# 5-Minute Demo Video Shooting Script

**Runtime target:** 5:00 (exactly, ± 5 seconds)
**Format:** Screen recording with voiceover
**Tool suggestion:** QuickTime Player (Mac, built-in) or OBS Studio (free), plus any mic that is not your laptop speakers.

---

## Pre-record checklist

1. **Rehearse once through** reading the script aloud with a stopwatch. Aim for ~140 words per minute.
2. **Warm up the hosted app** — open https://genaicapstone4-anot5dcxcx9vhuxzsifnzb.streamlit.app/ ~5 minutes before recording so it's not in the hibernation "wake up" screen.
3. **Pre-fill the AI Care Coordinator form** with this patient before you hit record:
   - Age: 53, Gender: Female, Lead Time: 47, SMS Reminder: ON, Hypertension: ON, Diabetes: ON, Scholarship: ON, Disability Level: 1, Previous No-Shows: 1.
4. **Open three browser tabs**, in this order, so you can Cmd+1/2/3 between them:
   - Tab 1 — GitHub repo: https://github.com/Ansh1816/gen_ai_capstone_4
   - Tab 2 — PDF report: open `report/report.pdf` (File → Open in Preview)
   - Tab 3 — Live Streamlit app
5. **Close every other window.** Disable notifications (macOS → Focus → Do Not Disturb).
6. **Record at 1080p minimum.** If using QuickTime: File → New Screen Recording → click the mic icon to pick your external mic → Start Recording.

---

## 0:00 – 0:10 · Opening title

**On screen:** GitHub repo page (Tab 1), scrolled to the top showing the README with the Streamlit badge.

> *"Hi, I'm Praanshu Ranjan. This is a 5-minute walkthrough of our end-semester capstone — an intelligent appointment no-show prediction and agentic care coordination assistant. The full code, the live app, and the project report are all in the repository on screen."*

---

## 0:10 – 0:45 · Quick code/deliverables checklist (35 sec)

**On screen:** Scroll slowly through the GitHub repo file list, hovering the cursor over each item as you mention it. Then switch to Tab 2 (the compiled report PDF) for the last 5 seconds.

> *"Here's the quick delivery checklist.*
> *One — the codebase. Streamlit front-end in app.py, the trained machine learning model in model.pkl, the agentic workflow in agent.py, the ReAct chat agent in chatbot_agent.py, and the vector store builder in build_vectorstore.py.*
> *Two — the knowledge base. Five hospital operations markdown files under the guidelines folder, and the pre-built ChromaDB vector index committed so the hosted app ships with a populated knowledge base.*
> *Three — the README documents setup, environment variables, and deployment. Four — the requirements file pins every Python dependency. Five — the live app is deployed to Streamlit Community Cloud. Six — the 23-page LaTeX project report lives in the report folder."*

[Cut to PDF in Preview showing the title page for a split second]

> *"All code, commits, and the report are on GitHub under our two-person team."*

---

## 0:45 – 1:00 · Problem statement (15 sec)

**On screen:** Switch to Tab 3, the live Streamlit app, but don't interact yet — just let the dark-blue header sit on screen.

> *"The problem. Around one in five patients miss their scheduled medical appointment. That costs the hospital money, blocks capacity from other patients, and breaks continuity of care. Our system does two things: it predicts who will miss, and it tells front-desk staff exactly what to do about it, grounded in the hospital's own rule-books."*

---

## 1:00 – 1:45 · Single Patient Analysis tab (45 sec)

**On screen:** Click the "SINGLE PATIENT ANALYSIS" tab. Set a patient — say Age 65, Female, Lead Time 30, toggle Hypertension and Diabetes ON. Click **Analyse Appointment Risk**.

> *"Tab one is the classical machine learning layer. I enter a patient's profile here on the left — age, gender, lead time, SMS reminder, chronic conditions, prior no-show history. The Decision Tree model takes fourteen features, including two we engineered: `AwaitingTime`, the days between booking and appointment, and `Num_App_Missed`, the patient's prior-miss count.*
> *I click Analyse, and on the right I get the probability, the risk tier — Low, Medium, or High — and a rule-based primary action. Below it is a scenario simulator: I can slide the lead time down and watch the risk fall in real time. Useful for front-desk staff negotiating an earlier slot with a patient."*

[Demonstrate sliding `Lead Time` down from 30 to 5 in the scenario panel, let the user see the delta badge update]

---

## 1:45 – 2:05 · Batch Analysis tab (20 sec)

**On screen:** Click the "BATCH ANALYSIS" tab.

> *"Tab two is for hospitals that want to run the model over a full day's schedule. Upload a CSV from the Hospital Management System, and you get KPI cards for total appointments, high-risk count, estimated no-shows, and suggested standby slots — plus a sortable high-risk patient list that you can export. I'll skip the upload to save time, but the logic is the same Decision Tree, just vectorised over every row of the CSV."*

---

## 2:05 – 3:05 · AI Care Coordinator — chat + RAG (60 sec)

**On screen:** Click the "AI CARE COORDINATOR" tab. Scroll to the chat input at the bottom of the right panel.

> *"Tab three is where this project stops being a mid-sem project and becomes an end-sem one. This is the agentic layer. I can chat freely with the Care Coordinator, and it has access to the machine learning model and the hospital rule-book as tools."*

[Type into the chat, or read from clipboard]: **"What is the reminder policy for high-risk patients? Cite the source document."**

[Press enter. Wait for the response to arrive.]

> *"Under the hood, the agent is using a LangGraph ReAct loop. It sees my question, decides which tool to call, and because I asked about policy, it calls the `search_guidelines` tool. That tool runs a similarity search over the ChromaDB vector store — forty chunks embedded with MiniLM — and returns the matching excerpts tagged with their source file names.*
> *Here's the answer. Notice it tells me the specific protocol — twenty-four hour pre-appointment call, four-hour confirmation deadline — and it cites `attendance_management.md` and `reminder_policy.md` by name. If I click 'View Agent Logic and Tool Usage' here, I can see which tools were called and what they returned. Every policy statement is traceable to a file."*

[Click the "View Agent Logic & Tool Usage" expander briefly to show it, then close]

---

## 3:05 – 4:30 · Run Full Care Workflow — the 5-step structured pipeline (85 sec)

**On screen:** On the left panel, scroll down to the ACTION OPTIONS card. The patient profile should already be set to Age 53, Female, 47-day lead time, hypertension + diabetes + scholarship, Disability 1, Prior No-Shows 1. Click **Run Full Care Workflow**.

> *"The chat is the free-form surface. The real architectural centrepiece is this button — Run Full Care Workflow. It executes a five-step LangGraph StateGraph with a strongly typed AgentState that flows through every node. Let me run it on a harder patient — fifty-three-year-old female, forty-seven-day lead time, hypertension and diabetes, on scholarship, one previous no-show. Hit run."*

[Wait ~45 seconds for the workflow to complete — spinner visible]

> *"Five steps. Step One — risk assessment — the Decision Tree fires and gives me the probability and tier. Step Two — an LLM call that analyses the patient's specific risk factors anchored to the actual values — see, it references the forty-seven-day lead time, the prior miss, and the scholarship flag specifically."*

[Scroll down through the expanded card, moving past Step 1, Step 2]

> *"Step Three is the one that makes this end-sem grade. The agent constructs a query from the patient's tier and factors, retrieves the top five chunks from the ChromaDB vector store, and displays them right here with source chips. You can see the five filenames — attendance_management.md, reminder_policy.md, patient_engagement.md, overbooking_guideline.md, ethical_guidelines.md — and below each chip, the exact excerpt that was retrieved."*

[Click Step 3 to expand it if it isn't already]

> *"Step Four takes those retrieved excerpts and the risk analysis from Step Two, and produces a structured intervention plan with a primary action, numbered strategies, a timeline, an escalation protocol, and an expected outcome. Every policy claim in the plan has a `[Source: filename]` tag. Step Five compiles the full care coordination report."*

[Scroll to the bottom to show the permanent Operational & Ethical Notice card]

> *"And beneath the five steps, a permanent ethical disclaimer that the AI is decision support only, every recommendation must be reviewed by qualified staff, and the system does not provide medical advice."*

---

## 4:30 – 4:50 · Guardrails + honest failure mode (20 sec)

**On screen:** Stay on the AI Care Coordinator tab.

> *"One thing I want to highlight. When we first deployed, the ChromaDB folder was excluded from Git, so the live agent had an empty vector store and was hallucinating fake hospital policies. I diagnosed that, committed the populated vector store, and hardened the agent's system prompt to refuse to invent policies and always cite a source. That entire incident and its fix is written up in section nine of the project report."*

---

## 4:50 – 5:00 · Closing (10 sec)

**On screen:** Switch back to Tab 1 (GitHub repo) briefly, then back to the app.

> *"The code, the live app, the report, and this video are all linked from the repository README. Thanks for watching."*

---

## Post-record checklist

1. **Trim** the recording to exactly 5 minutes — remove any dead air at the very start and end.
2. **Normalise audio** — use any free tool (Audacity, DaVinci Resolve) to level out the volume so soft and loud sections are balanced.
3. **Export as MP4 H.264**, 1080p, ~30 fps. Target file size under 200 MB for easy sharing.
4. **Upload to:**
   - **YouTube (unlisted)** or **Google Drive** (easy), OR
   - **Vimeo** (nicer quality, still free tier)
5. **Add the URL** to the top of the GitHub README as a badge:
   ```
   [![Demo Video](https://img.shields.io/badge/Demo-5%20min-blue?logo=youtube)](YOUR_VIDEO_URL)
   ```

---

## Pacing tips (important)

- The script is **≈700 words** at a normal speaking pace (~140 wpm) — that lands right at 5:00.
- **Breathe between sentences.** It feels slow to you, but a grader hears it as confident.
- If any section runs long in rehearsal, **cut Batch Analysis first** (it's the least distinctive tab) — you can drop that 20-second block entirely and compensate by slowing down the workflow demo.
- **Don't click fast.** The grader is trying to read what's on screen while you talk. Slow cursor movement and a 1–2 second pause on each screen they need to read.

---

## If things go wrong during recording

- **Spinner stalls on Run Full Care Workflow:** click the browser refresh, redo the patient, try again. The free-tier Groq API occasionally throttles.
- **App is hibernating:** the "Yes, get this app back up!" screen appears — click it, wait, then restart the recording from the beginning of that section.
- **You stutter a word:** just pause for 2 seconds and say the sentence again. You'll edit out the bad take cleanly because the pause creates a visible gap in the audio waveform.
