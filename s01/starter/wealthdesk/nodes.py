"""
wealthdesk/nodes.py
-------------------
Node functions for the WealthDesk graph.

Each node is a plain Python function:
  - Input : the full WealthDeskState (read-only)
  - Output: a dict containing ONLY the keys this node changed
             (LangGraph merges it into the state automatically)
"""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .config import DECLINE_RESPONSE, ESCALATE_RESPONSE, SYSTEM_PROMPT, CLASSIFY_SYSTEM_PROMPT
from .state import WealthDeskState
from .tools import llm, classifier_llm


BLOCKLIST = [
    "ignore all previous",
    "forget everything",
    "you are now",
    "disregard your system",
    "act as",
    "jailbreak",
]


def classify(state: WealthDeskState) -> dict:
    """Call the LLM and return the agent's reply."""

    msg = state["customer_message"].strip()
 
    if any(phrase in msg.lower() for phrase in BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE"}
 
    if not msg or len(msg) < 10 or len(msg) > 500:
        return {"query_type": "OUT_OF_SCOPE"}
    
    messages = [
      SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
      HumanMessage(content=state["customer_message"])
    ]

    try:
        result =  classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in ["SIMPLE", "COMPLEX", "Out of Scope"]:
            query_type = "SIMPLE"  # Default to SIMPLE if the classification is unexpected
    except Exception as e:
        print(f"[WealthDesk] Error occurred: {e}")
        query_type = "SIMPLE"
    
    return {"query_type": query_type}


def escalate(state: WealthDeskState) -> dict:
    """Escalate the query to a human agent."""
    new_history = state.get("history", []) + [
        {"customer_message": state["customer_message"], "response": ESCALATE_RESPONSE}
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}

def decline(state: WealthDeskState) -> dict:
    """Decline the query and provide a polite response."""
    new_history = state.get("history", []) + [
        {"customer_message": state["customer_message"], "response": DECLINE_RESPONSE}
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: WealthDeskState) -> dict:
    """Route the query based on its classification."""
    query_type = state.get("query_type", "SIMPLE")
    if query_type == "COMPLEX":
        return "escalate"
    if query_type == "OUT_OF_SCOPE":
        return "decline"
    # Default to SIMPLE if the classification is unexpected
    return "respond"


def respond(state: WealthDeskState) -> dict:
    """Call the LLM and return the agent's reply."""
    messages = [
      SystemMessage(content=SYSTEM_PROMPT)
    ]
    history = state.get("history", [])
    for turn in history:
        messages.append(HumanMessage(content=turn["customer_message"]))
        messages.append(AIMessage(content=turn["response"]))

    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm.invoke(messages)
        response_text = result.content.strip()
    except Exception as e:
        print(f"[WealthDesk] Error occurred: {e}")
        return {"response": "I'm sorry, but I encountered an error while processing your request."}
    
    new_history = history + [{"customer_message": state["customer_message"], "response": response_text}]
    return {"response": response_text, "history": new_history}
