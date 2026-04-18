"""
agent.py
--------
LangGraph-based Agentic Care Coordination Assistant.

Nodes:
  1. risk_assessment   — ML model predicts no-show probability
  2. risk_reasoning    — LLM analyses risk factors
  3. guideline_retrieval — RAG from ChromaDB
  4. intervention_plan — LLM generates structured intervention
  5. report_generation — assembles final structured report
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

import joblib
import pandas as pd
import numpy as np

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

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

# ── LLM ──────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Embeddings & Vector Store ────────────────────────────────────────
_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
_vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=_embeddings,
)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})


# ═══════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    # inputs
    patient_data: dict          # age, gender, awaiting, sms, hipert, diab, schol, hcap, prev_miss
    # after risk_assessment
    risk_score: float
    risk_tier: str
    risk_color: str
    # after risk_reasoning
    risk_analysis: str
    risk_factors: list
    # after guideline_retrieval
    retrieved_guidelines: list  # list of {"content": ..., "source": ...}
    # after intervention_plan
    intervention_plan: str
    # after report_generation
    final_report: str


# ═══════════════════════════════════════════════════════════════════════
# NODE 1: RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════
def risk_assessment(state: AgentState) -> dict:
    """Use the trained ML model to predict no-show probability."""
    p = state["patient_data"]

    # Build feature row — same logic as app.py predict_prob()
    h = {f"Handicap_{i}": 0 for i in range(5)}
    h[f"Handicap_{p['hcap']}"] = 1
    row = {
        "Age": p["age"],
        "AwaitingTime": p["awaiting"],
        "SMS_received": 1 if p["sms"] else 0,
        "Hipertension": 1 if p["hipert"] else 0,
        "Diabetes": 1 if p["diab"] else 0,
        "Scholarship": 1 if p["schol"] else 0,
        "Num_App_Missed": p["prev_miss"],
        "Gender_F": 1 if p["gender"] == "Female" else 0,
        "Gender_M": 1 if p["gender"] == "Male" else 0,
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

    return {"risk_score": prob, "risk_tier": tier, "risk_color": color}


# ═══════════════════════════════════════════════════════════════════════
# NODE 2: RISK REASONING
# ═══════════════════════════════════════════════════════════════════════
_risk_reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a healthcare operations AI analyst. Given a patient's profile and their predicted no-show probability, analyse the key risk factors contributing to this prediction.

Be specific, data-driven, and professional. Reference the patient's actual values.
Respond in this format:

RISK FACTOR ANALYSIS:
- [factor 1]: [explanation]
- [factor 2]: [explanation]
- [factor 3]: [explanation]

OVERALL ASSESSMENT:
[2-3 sentence summary of why this patient is at this risk level]"""),
    ("human", """Patient Profile:
- Age: {age}
- Gender: {gender}
- Lead Time (days between booking and appointment): {awaiting} days
- SMS Reminder Sent: {sms}
- Hypertension: {hipert}
- Diabetes: {diab}
- Scholarship/Social Insurance: {schol}
- Disability Level: {hcap}
- Previous No-Shows: {prev_miss}

Predicted No-Show Probability: {risk_score:.1%}
Risk Tier: {risk_tier}

Analyse the risk factors for this patient."""),
])

def risk_reasoning(state: AgentState) -> dict:
    """LLM analyses the patient's risk factors."""
    p = state["patient_data"]
    chain = _risk_reasoning_prompt | llm | StrOutputParser()
    analysis = chain.invoke({
        "age": p["age"],
        "gender": p["gender"],
        "awaiting": p["awaiting"],
        "sms": "Yes" if p["sms"] else "No",
        "hipert": "Yes" if p["hipert"] else "No",
        "diab": "Yes" if p["diab"] else "No",
        "schol": "Yes" if p["schol"] else "No",
        "hcap": p["hcap"],
        "prev_miss": p["prev_miss"],
        "risk_score": state["risk_score"],
        "risk_tier": state["risk_tier"],
    })

    # Extract factors as a list
    factors = []
    for line in analysis.split("\n"):
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            factors.append(line[2:])

    return {"risk_analysis": analysis, "risk_factors": factors}


# ═══════════════════════════════════════════════════════════════════════
# NODE 3: GUIDELINE RETRIEVAL (RAG)
# ═══════════════════════════════════════════════════════════════════════
def guideline_retrieval(state: AgentState) -> dict:
    """Retrieve relevant healthcare guidelines from ChromaDB."""
    query = (
        f"Patient with {state['risk_tier']} risk of appointment no-show. "
        f"No-show probability {state['risk_score']:.0%}. "
        f"Key factors: {', '.join(state.get('risk_factors', [])[:3])}. "
        f"What are the recommended intervention strategies and operational guidelines?"
    )

    docs = _retriever.invoke(query)

    guidelines = []
    seen = set()
    for doc in docs:
        content = doc.page_content.strip()
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        # Deduplicate
        if content not in seen:
            seen.add(content)
            guidelines.append({"content": content, "source": source})

    return {"retrieved_guidelines": guidelines}


# ═══════════════════════════════════════════════════════════════════════
# NODE 4: INTERVENTION PLAN
# ═══════════════════════════════════════════════════════════════════════
_intervention_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a healthcare care coordination AI assistant. Based on the patient's risk assessment and retrieved operational guidelines, create a structured intervention plan.

Your response MUST follow this exact format:

PRIMARY ACTION: [Call Patient / Send SMS Reminder / Standard Confirmation]

INTERVENTION STRATEGIES:
1. [Strategy name]: [Specific actionable step]
2. [Strategy name]: [Specific actionable step]
3. [Strategy name]: [Specific actionable step]

TIMELINE:
- [Timeframe]: [Action to take]
- [Timeframe]: [Action to take]

ESCALATION PROTOCOL:
[What to do if initial intervention fails]

EXPECTED OUTCOME:
[Brief prediction of intervention effectiveness]"""),
    ("human", """Patient Risk Assessment:
- No-Show Probability: {risk_score:.1%}
- Risk Tier: {risk_tier}

Risk Factor Analysis:
{risk_analysis}

Retrieved Operational Guidelines:
{guidelines_text}

Create a structured intervention plan for this patient."""),
])

def intervention_plan(state: AgentState) -> dict:
    """LLM generates a structured intervention plan using risk analysis + guidelines."""
    guidelines_text = "\n\n".join(
        f"[Source: {g['source']}]\n{g['content']}"
        for g in state.get("retrieved_guidelines", [])
    )
    if not guidelines_text:
        guidelines_text = "No specific guidelines retrieved."

    chain = _intervention_prompt | llm | StrOutputParser()
    plan = chain.invoke({
        "risk_score": state["risk_score"],
        "risk_tier": state["risk_tier"],
        "risk_analysis": state.get("risk_analysis", ""),
        "guidelines_text": guidelines_text,
    })
    return {"intervention_plan": plan}


# ═══════════════════════════════════════════════════════════════════════
# NODE 5: REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════
_report_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a healthcare AI report generator. Compile all the information into a professional, structured care coordination report.

Use this format:

═══════════════════════════════════════
 CARE COORDINATION REPORT
═══════════════════════════════════════

PATIENT SUMMARY
• Age: [age] | Gender: [gender]
• Lead Time: [days] days | SMS Sent: [yes/no]
• Conditions: [list conditions]
• Prior No-Shows: [count]

RISK ASSESSMENT
• No-Show Probability: [X%]
• Risk Tier: [LOW/MEDIUM/HIGH]

RISK FACTOR ANALYSIS
[Include the analysis from risk reasoning]

INTERVENTION PLAN
[Include the structured intervention plan]

SUPPORTING GUIDELINES
[List retrieved guidelines with source citations]

ETHICAL DISCLAIMER
This report is generated by an AI-powered decision support system. All recommendations are probabilistic and based on historical data patterns. Interventions must be reviewed and approved by qualified healthcare administrative staff before implementation. This system does not provide medical advice. Patient privacy and data governance regulations apply to all actions taken based on this report."""),
    ("human", """Compile the care coordination report with these inputs:

Patient Data:
- Age: {age}, Gender: {gender}
- Lead Time: {awaiting} days, SMS Sent: {sms}
- Hypertension: {hipert}, Diabetes: {diab}, Scholarship: {schol}
- Disability Level: {hcap}
- Previous No-Shows: {prev_miss}

Risk Score: {risk_score:.1%}
Risk Tier: {risk_tier}

Risk Analysis:
{risk_analysis}

Intervention Plan:
{intervention_plan}

Retrieved Guidelines (Sources):
{guidelines_sources}"""),
])

def report_generation(state: AgentState) -> dict:
    """Generate the final structured report."""
    p = state["patient_data"]
    guidelines_sources = "\n".join(
        f"- [{g['source']}]: {g['content'][:120]}..."
        for g in state.get("retrieved_guidelines", [])
    )
    if not guidelines_sources:
        guidelines_sources = "No specific guidelines retrieved."

    chain = _report_prompt | llm | StrOutputParser()
    report = chain.invoke({
        "age": p["age"],
        "gender": p["gender"],
        "awaiting": p["awaiting"],
        "sms": "Yes" if p["sms"] else "No",
        "hipert": "Yes" if p["hipert"] else "No",
        "diab": "Yes" if p["diab"] else "No",
        "schol": "Yes" if p["schol"] else "No",
        "hcap": p["hcap"],
        "prev_miss": p["prev_miss"],
        "risk_score": state["risk_score"],
        "risk_tier": state["risk_tier"],
        "risk_analysis": state.get("risk_analysis", ""),
        "intervention_plan": state.get("intervention_plan", ""),
        "guidelines_sources": guidelines_sources,
    })
    return {"final_report": report}


# ═══════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════════
def build_graph():
    """Build and compile the LangGraph care coordination workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("risk_assessment", risk_assessment)
    graph.add_node("risk_reasoning", risk_reasoning)
    graph.add_node("guideline_retrieval", guideline_retrieval)
    graph.add_node("intervention_plan", intervention_plan)
    graph.add_node("report_generation", report_generation)

    graph.set_entry_point("risk_assessment")
    graph.add_edge("risk_assessment", "risk_reasoning")
    graph.add_edge("risk_reasoning", "guideline_retrieval")
    graph.add_edge("guideline_retrieval", "intervention_plan")
    graph.add_edge("intervention_plan", "report_generation")
    graph.add_edge("report_generation", END)

    return graph.compile()


# Pre-compiled graph
care_coordination_graph = build_graph()


def run_agent(patient_data: dict) -> dict:
    """
    Run the full care coordination agent.

    Args:
        patient_data: dict with keys:
            age, gender, awaiting, sms, hipert, diab, schol, hcap, prev_miss

    Returns:
        Final state dict with all agent outputs.
    """
    initial_state = {
        "patient_data": patient_data,
        "risk_score": 0.0,
        "risk_tier": "",
        "risk_color": "",
        "risk_analysis": "",
        "risk_factors": [],
        "retrieved_guidelines": [],
        "intervention_plan": "",
        "final_report": "",
    }
    result = care_coordination_graph.invoke(initial_state)
    return result
