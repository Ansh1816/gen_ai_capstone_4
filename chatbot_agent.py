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
    """Predicts the likelihood of a patient missing their appointment (no-show risk).
    Pass in the patient's demographic and historical criteria. Use 'yes' or 'no' for conditions.
    Returns the numeric risk score and the risk tier (LOW, MEDIUM, or HIGH).
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
    """Searches the hospital operational guidelines knowledge base for protocols and best practices.
    Pass in a descriptive search query (e.g. 'what to do for high risk patients' or 'SMS reminder policy').
    Returns excerpts from the official documentation.
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
    "You are an AI Care Coordination Advisor. "
    "Your goal is to help healthcare staff analyze patient no-show risks, explain the factors, "
    "and recommend actionable intervention strategies based on hospital guidelines. "
    "Whenever asked about a patient, you MUST first use the `predict_noshow` tool. "
    "After getting the risk, you should use the `search_guidelines` tool to see what policies apply to that risk tier. "
    "Finally, summarize everything clearly to the user in a friendly, professional manner. "
    "If a user asks for a report, provide the content in a structured markdown format."
))

agent_executor = create_react_agent(llm, tools, prompt=system_message)
