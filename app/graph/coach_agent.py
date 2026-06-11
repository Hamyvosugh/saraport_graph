"""Coach Agent — the main health coaching brain using LangChain ReAct agent.

Uses DeepSeek LLM + tools (from tools.py) to:
1. Read user profile, plan, logs, memory, chat history
2. Analyze food, log progress, save memory/messages
3. Return helpful Persian coaching replies
"""

import os
from typing import Optional
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Optional imports - gracefully degrade if not installed
# langgraph >=0.3 moved prebuilt to separate package "langgraph-prebuilt"
create_react_agent = None
CompiledStateGraph = None
_LANGGRAPH_AVAILABLE = False

try:
    from langgraph.prebuilt import create_react_agent  # noqa: F811
    from langgraph.graph.state import CompiledStateGraph  # noqa: F811
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    try:
        from langgraph_prebuilt import create_react_agent  # noqa: F811
        from langgraph.graph.state import CompiledStateGraph  # noqa: F811
        _LANGGRAPH_AVAILABLE = True
    except ImportError:
        pass

try:
    from langchain_deepseek import ChatDeepSeek
    _DEEPSEEK_AVAILABLE = True
except ImportError:
    _DEEPSEEK_AVAILABLE = False
    ChatDeepSeek = None

from app.graph.tools import ALL_TOOLS
from app.memory.supabase_store import store as db

load_dotenv(dotenv_path=".env.local", override=False)
load_dotenv(dotenv_path=".env", override=False)

# ---------------------------------------------------------------------------
# Persian/English system prompt for the health coach
# ---------------------------------------------------------------------------
COACH_SYSTEM_PROMPT = """You are a professional health, nutrition, and fitness coach named "Sara" (سارا). 
You are warm, encouraging, knowledgeable, and supportive. You communicate primarily in Persian (فارسی).

## Your Role
You help users achieve their health goals through:
- Daily coaching conversations
- Reviewing their logs (weight, steps, water, sleep)
- Analyzing their food intake
- Suggesting improvements to their daily plan
- Remembering past conversations to provide personalized advice

## Important Rules
1. ALWAYS respond in Persian (فارسی) unless the user writes in English
2. Be concise — keep replies to 3-5 sentences unless a detailed analysis is requested
3. Be encouraging and positive — celebrate small wins
4. Use emojis sparingly (1-2 max per message)
5. When the user shares what they ate, use the analyze_food_note tool
6. When the user shares metrics (steps, weight, water, sleep), use the log_daily_progress tool
7. Before giving advice, check the user's profile, plan, and recent logs using the appropriate tools
8. After each meaningful conversation turn, save a summary to memory using save_coach_memory
9. NEVER ask the user for their user_id — it is automatically provided to you
10. If you don't have enough information to answer, ask the user for the missing details

## Tools Available
- get_user_profile: Read user health profile
- get_user_plan: Read daily plan/targets
- get_user_logs: Read recent daily logs
- get_user_memory: Read conversation memory
- get_chat_history: Read recent chat messages
- save_coach_memory: Save conversation summary to memory
- analyze_food_note: Analyze food description for calories/macros
- log_daily_progress: Save daily metrics (steps, water, sleep, weight)
- save_message: Save a message to chat history (you don't need to call this directly)

## Your Workflow
When a user sends a message:
1. First, understand what they're saying/asking
2. Use tools to gather relevant context (profile, plan, logs, history)
3. If they shared food → analyze it
4. If they shared metrics → log them  
5. Provide a helpful, personalized response
6. Save important context to memory for future conversations

You are the core intelligence of Saraport. Be the best health coach you can be!
"""


def _get_llm():
    """Create DeepSeek LLM instance, with graceful fallback."""
    if not _DEEPSEEK_AVAILABLE:
        raise ImportError("langchain-deepseek package not installed")
    return ChatDeepSeek(model="deepseek-chat", temperature=0.7)


def build_coach_agent():
    """Build a LangGraph prebuilt ReAct agent with coaching tools."""
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph.prebuilt package not installed")
    model = _get_llm()
    return create_react_agent(model=model, tools=ALL_TOOLS)


# ---------------------------------------------------------------------------
# Singleton agent (lazy init)
# ---------------------------------------------------------------------------
_agent: object = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_coach_agent()
    return _agent


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_coach_agent(user_id: str, message: str) -> dict:
    """Run the coach agent for a user message.

    Args:
        user_id: The Supabase user ID
        message: The user's chat message

    Returns:
        {"reply": str, "action": str, "data": dict}
    """
    if not _LANGGRAPH_AVAILABLE or not _DEEPSEEK_AVAILABLE:
        raise ImportError(
            "Coach agent requires langgraph.prebuilt and langchain-deepseek. "
            "Install with: pip install langgraph-prebuilt langchain-deepseek"
        )

    # 1. Load recent chat history for context
    chat_msgs = db.get_messages(user_id, limit=20)
    history_str = ""
    if chat_msgs:
        history_parts = []
        for m in chat_msgs[-10:]:  # last 10 messages
            role_label = "کاربر" if m.get("role") == "user" else "مربی"
            history_parts.append(f"{role_label}: {m.get('content', '')}")
        history_str = "\n".join(history_parts)
        history_str = f"Previous conversation:\n{history_str}\n"

    # 2. Pre-inject user_id into the message so tools can use it
    enriched_message = f"[user_id={user_id}] {message}"

    # 3. Invoke the agent
    agent = get_agent()
    result = {}
    reply = ""
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=f"{COACH_SYSTEM_PROMPT}\n\n{history_str}\n\nUser message: {enriched_message}")],
        })
        # Extract the last AI message as the reply
        msgs = result.get("messages", [])
        for m in reversed(msgs):
            if hasattr(m, "content") and m.type == "ai":
                reply = m.content
                break
        if not reply and msgs:
            reply = str(msgs[-1].content) if hasattr(msgs[-1], "content") else str(msgs[-1])
    except Exception as e:
        reply = f"⚠️ خطا در ارتباط با هوش مصنوعی. لطفاً دوباره تلاش کنید.\n\n(Error: {str(e)[:200]})"

    # 4. Save messages to Supabase (non-critical)
    try:
        db.add_message(user_id, "user", message)
        if reply:
            db.add_message(user_id, "assistant", reply)
    except Exception:
        pass  # DB save failure shouldn't block the reply

    # 5. Determine action type based on tool calls in messages
    action = "chat"
    data = {}
    result_str = str(result.get("messages", []))
    if "analyze_food_note" in result_str:
        action = "food_analysis"
    elif "log_daily_progress" in result_str:
        action = "daily_log"

    return {
        "reply": reply,
        "action": action,
        "data": data,
    }
