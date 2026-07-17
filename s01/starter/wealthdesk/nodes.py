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

from .config import SYSTEM_PROMPT
from .state import WealthDeskState
from .tools import llm


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
