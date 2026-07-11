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


# ---------------------------------------------------------------------------
# TODO 4 of 5 -- respond node
# ---------------------------------------------------------------------------
# Implement the respond() function so it:
#
#   1. Builds a messages list:
#        messages = [
#            SystemMessage(content=SYSTEM_PROMPT),
#            HumanMessage(content=state["customer_message"]),
#        ]
#
#   2. Calls the LLM inside a try / except block:
#        result = llm.invoke(messages)
#
#   3. On success  → return {"response": result.content}
#      On exception → print the error with a [WealthDesk] prefix
#                      and return a safe fallback string so the
#                      agent never crashes mid-conversation.
#
# ---------------------------------------------------------------------------

def respond(state: WealthDeskState) -> dict:
    """Call the LLM and return the agent's reply."""
    messages = [
      SystemMessage(content=SYSTEM_PROMPT)
    ]
    history = state.get("history", [])
    for turn in history:
        messages.append(HumanMessage(content=turn["customer_message"]))
    else:
        messages.append(AIMessage(content=state["content"]))

    messages.append(HumanMessage(content=state["customer_message"]))  

    try:
        result = llm.invoke(messages)
        response_text = result.content.strip()
    except Exception as e:
        print(f"[WealthDesk] Error occurred: {e}")
        return {"response": "I'm sorry, but I encountered an error while processing your request."}
    
    new_history = history + [{"role": "user", "content": state["customermessage"]},
                            {"role": "assistant", "content": response_text}]
    return {"response": response_text, "history": new_history}
