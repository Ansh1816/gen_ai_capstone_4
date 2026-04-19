"""
chatbot_agent.py
----------------
LangGraph-based ReAct Agent for Care Coordination.
Uses tools to interact with ML models and ChromaDB.
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

import joblib
import pandas as pd
import numpy as np

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_cols.pkl")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── Load ML artifacts ────────────────────────────────────────────────
_model = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)
_feature_cols = joblib.load(FEATURES_PATH)

# ── Embeddings & Vector Store ────────────────────────────────────────
_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
_vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=_embeddings,
)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

# ── LLM ──────────────────────────────────────────────────────────────
# We use Mixtral or Llama-3, Groq supports tool calling very well on llama-3.1-8b-instant
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ═══════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════

@tool
def predict_noshow(age: int, gender: str, awaiting_days: int, sms_sent: str,
                   hypertension: str, diabetes: str, scholarship: str,
                   disability_level: int, previous_no_shows: int) -> dict:
    """Runs the trained ML model and returns this patient's no-show probability and tier.
    Call this FIRST for any question that mentions a specific patient, even if the user
    has not asked for a prediction — downstream advice depends on the tier.

    Inputs: age (int), gender ('Male'/'Female'), awaiting_days (int, days between booking
    and appointment), sms_sent/hypertension/diabetes/scholarship ('yes' or 'no'),
    disability_level (0-4), previous_no_shows (int).

    Returns: {"risk_score": float in [0,1], "risk_tier": "LOW" | "MEDIUM" | "HIGH"}.
    Always surface the exact percentage to the user, never just the tier.
    """
    def is_yes(v: str) -> int:
        return 1 if str(v).lower() in ["yes", "true", "1", "y"] else 0
        
    h = {f"Handicap_{i}": 0 for i in range(5)}
    h[f"Handicap_{disability_level}"] = 1
    row = {
        "Age": age,
        "AwaitingTime": awaiting_days,
        "SMS_received": is_yes(sms_sent),
        "Hipertension": is_yes(hypertension),
        "Diabetes": is_yes(diabetes),
        "Scholarship": is_yes(scholarship),
        "Num_App_Missed": previous_no_shows,
        "Gender_F": 1 if gender.lower() == "female" else 0,
        "Gender_M": 1 if gender.lower() == "male" else 0,
        **h,
    }
    df_row = pd.DataFrame([row])
    for col in _feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[_feature_cols]
    scaled = _scaler.transform(df_row)
    prob = float(_model.predict_proba(scaled)[0][1])

    if prob < 0.30:
        tier, color = "LOW", "#16a34a"
    elif prob < 0.55:
        tier, color = "MEDIUM", "#f59e0b"
    else:
        tier, color = "HIGH", "#dc2626"

    return {"risk_score": prob, "risk_tier": tier}

@tool
def search_guidelines(query: str) -> str:
    """Searches the hospital's operational guidelines library and returns real excerpts.
    These are the ONLY authoritative sources of policy in this system — if an answer is
    not grounded in what this tool returns, it must not be presented as a hospital rule.

    Pass a descriptive query such as 'high-risk patient phone protocol', 'SMS reminder
    timing', 'overbooking standby rules', or 'chronic condition escalation'.

    Returns one or more excerpts, each prefixed with [Source: <filename>]. If the tool
    returns 'No specific guidelines found for this query.', you MUST tell the user
    plainly that no matching policy was found — do NOT invent policies or fall back to
    generic healthcare advice.
    """
    docs = _retriever.invoke(query)
    results = []
    for d in docs:
        source = os.path.basename(d.metadata.get("source", "Unknown"))
        results.append(f"[Source: {source}]\n{d.page_content.strip()}")

    if not results:
        return "No specific guidelines found for this query."
    return "\n\n---\n\n".join(results)


# ═══════════════════════════════════════════════════════════════════════
# AGENT SETUP
# ═══════════════════════════════════════════════════════════════════════
tools = [predict_noshow, search_guidelines]

system_message = SystemMessage(content=(
    "You are the AI Care Coordination Advisor for a hospital's appointment-attendance "
    "operations team. You help staff assess patient no-show risk and recommend "
    "interventions that are grounded in the hospital's own written guidelines.\n\n"

    "WORKFLOW for any patient question:\n"
    "1. Call `predict_noshow` FIRST with the patient's profile and report the exact "
    "   numeric probability AND the risk tier (e.g. '47.3% — MEDIUM').\n"
    "2. Call `search_guidelines` at least once with a query that targets the tier and "
    "   the patient's specific factors (chronic conditions, long lead time, prior "
    "   no-shows, etc.). You may call it multiple times with different phrasings.\n"
    "3. Build the intervention plan STRICTLY from the excerpts the tool returned.\n\n"

    "CITATION & HALLUCINATION RULES (strict):\n"
    "• Any hospital rule, threshold, timing, or protocol you state MUST be backed by a "
    "  `[Source: <filename>]` citation from `search_guidelines` output. Put the filename "
    "  in brackets next to the claim — e.g. 'Place a phone call ≥24 hours before the "
    "  appointment [Source: attendance_management.md].'\n"
    "• If `search_guidelines` returns 'No specific guidelines found for this query.', "
    "  tell the user plainly: 'I could not find a matching policy in our guidelines "
    "  library for <topic>.' Do NOT invent policies. Do NOT fall back to generic "
    "  healthcare advice presented as hospital rules. You may suggest escalation to "
    "  a human care coordinator.\n"
    "• When asked to quote a document, quote EXACTLY from the tool output. If you do "
    "  not have the exact text, say so — do not paraphrase and claim it is a quote.\n\n"

    "OUTPUT FORMAT for a full patient analysis:\n"
    "**Risk:** <probability%> — <TIER>\n"
    "**Key Risk Factors:** short bullets referencing the patient's actual values.\n"
    "**Recommended Intervention:** numbered steps, each ending with a [Source: …] tag.\n"
    "**Sources Consulted:** comma-separated list of filenames cited above.\n"
    "**Disclaimer:** 'This is an AI-generated decision-support output based on "
    "historical appointment data and the hospital's written guidelines. Review with "
    "qualified administrative staff before taking operational action. This system does "
    "not provide medical advice.'\n\n"

    "If the user asks for a report, use the same structure in markdown. Keep answers "
    "concise and professional. Never repeat the patient's input back as the body of a "
    "plan — always give concrete, cited actions."
))

agent_executor = create_react_agent(llm, tools, prompt=system_message)
