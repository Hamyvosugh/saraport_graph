"""LangGraph workflow builder."""

from langgraph.graph import StateGraph, END
from app.graph.state import HealthState
from app.graph.nodes import (
    intake_node,
    planner_node,
    coach_node,
    food_vision_node,
    progress_node,
    memory_node,
    food_breakdown_node,
    food_breakdown_resolve_node,
)
from app.memory.store import store

# Optional import: coach_agent may fail if langgraph packages aren't installed
try:
    from app.graph.coach_agent import run_coach_agent
    _COACH_AGENT_AVAILABLE = True
except ImportError:
    _COACH_AGENT_AVAILABLE = False
    run_coach_agent = None


def build_graph() -> StateGraph:
    """Build and return a compiled StateGraph."""
    graph = StateGraph(HealthState)

    # Add nodes
    graph.add_node("intake", intake_node)
    graph.add_node("planner", planner_node)
    graph.add_node("coach", coach_node)
    graph.add_node("food_vision", food_vision_node)
    graph.add_node("progress", progress_node)
    graph.add_node("memory", memory_node)
    graph.add_node("food_breakdown", food_breakdown_node)

    # Edges: depends on the workflow type
    # We'll route conditionally in the compiled graph

    return graph


def run_onboarding(user_input: dict) -> dict:
    """Run the onboarding flow: intake → planner → END"""
    graph = build_graph()
    graph.add_edge("intake", "planner")
    graph.set_entry_point("intake")
    graph.add_edge("planner", END)

    app = graph.compile()
    state: HealthState = {
        "user_id": user_input.get("user_id", ""),
        "input": user_input,
        "messages": [],
        "profile": {},
        "plan": {},
        "logs": [],
        "memory": {},
        "output": {},
    }
    result = app.invoke(state)

    profile = store.get_profile(state["user_id"])
    plan = store.get_plan(state["user_id"])

    summary = (
        f"Profile: {profile.get('age', '?')}yo {profile.get('gender','?')}, "
        f"{profile.get('height_cm','?')}cm, {profile.get('weight_kg','?')}kg. "
        f"Goal: {profile.get('goal','')}. "
        f"Plan: {plan.get('calories','?')} kcal/day, "
        f"{plan.get('protein_g','?')}g protein."
    )
    return {
        "user_id": state["user_id"],
        "summary": summary,
        "initial_plan": state["messages"][-1]["content"] if state["messages"] else "",
        "daily_targets": plan,
    }


def run_chat(user_input: dict) -> dict:
    """Run the daily coach chat using the new CoachAgent (ReAct + tools).
    Falls back to simple coach_node if coach_agent is unavailable."""
    if not _COACH_AGENT_AVAILABLE:
        user_id = user_input.get("user_id", "")
        message = user_input.get("message", "")
        graph = build_graph()
        graph.set_entry_point("intake")
        graph.add_edge("intake", "coach")
        graph.add_edge("coach", END)
        app = graph.compile()
        state: HealthState = {
            "user_id": user_id,
            "input": {"message": message},
            "messages": [],
            "profile": store.get_profile(user_id),
            "plan": store.get_plan(user_id),
            "logs": store.get_logs(user_id),
            "memory": store.get_memory(user_id),
            "output": {},
        }
        result = app.invoke(state)
        return result.get("output", {"reply": "Service temporarily unavailable."})
    user_id = user_input.get("user_id", "")
    message = user_input.get("message", "")
    return run_coach_agent(user_id, message)


def run_coach(user_input: dict) -> dict:
    """Run the full coach agent flow with Supabase persistence.
    Falls back to simple coach_node if coach_agent is unavailable.

    Input: {"user_id": "...", "message": "..."}
    Output: {"reply": "...", "action": "...", "data": {...}}
    """
    if not _COACH_AGENT_AVAILABLE:
        return run_chat(user_input)
    user_id = user_input.get("user_id", "")
    message = user_input.get("message", "")
    return run_coach_agent(user_id, message)


def run_daily_log(user_input: dict) -> dict:
    """Run the daily log flow: intake → progress → memory → END"""
    graph = build_graph()
    graph.set_entry_point("intake")
    graph.add_edge("intake", "progress")
    graph.add_edge("progress", "memory")
    graph.add_edge("memory", END)

    app = graph.compile()
    state: HealthState = {
        "user_id": user_input.get("user_id", ""),
        "input": user_input,
        "messages": [],
        "profile": store.get_profile(user_input.get("user_id", "")),
        "plan": {},
        "logs": [],
        "memory": {},
        "output": {},
    }
    result = app.invoke(state)
    return result.get("output", {})


def run_food_analysis(user_input: dict) -> dict:
    """Run the food analysis flow: intake → food_vision → END"""
    graph = build_graph()
    graph.add_edge("intake", "food_vision")
    graph.set_entry_point("intake")
    graph.add_edge("food_vision", END)

    app = graph.compile()
    state: HealthState = {
        "user_id": user_input.get("user_id", ""),
        "input": user_input,
        "messages": [],
        "profile": {},
        "plan": {},
        "logs": [],
        "memory": {},
        "output": {},
    }
    result = app.invoke(state)
    return result.get("output", {})


def run_food_breakdown(user_input: dict) -> dict:
    """Run the AI food breakdown flow: intake → food_breakdown → END
    
    Input: {"user_id": "...", "text": "what the user ate"}
    Output: auto-saves to DB, returns summary or pending_questions
    """
    graph = build_graph()
    graph.set_entry_point("intake")
    graph.add_edge("intake", "food_breakdown")
    graph.add_edge("food_breakdown", END)

    app = graph.compile()
    state: HealthState = {
        "user_id": user_input.get("user_id", ""),
        "input": user_input,
        "messages": [],
        "profile": {},
        "plan": {},
        "logs": [],
        "memory": {},
        "output": {},
    }
    result = app.invoke(state)
    output = result.get("output", {})
    # If there are pending questions, include them
    if result.get("pending_questions"):
        output["pending_questions"] = result.get("pending_questions")
    return output


def run_food_breakdown_resolve(user_input: dict) -> dict:
    """Resolve pending food ambiguity from user's choices.
    
    Input: {"user_id": "...", "choices": [{"food_index": 0, "selected_food_id": "uuid"}], "text": "..."}
    """
    graph = build_graph()
    graph.add_node("food_breakdown_resolve", food_breakdown_resolve_node)
    graph.set_entry_point("intake")
    graph.add_edge("intake", "food_breakdown_resolve")
    graph.add_edge("food_breakdown_resolve", END)

    app = graph.compile()
    state: HealthState = {
        "user_id": user_input.get("user_id", ""),
        "input": user_input,
        "messages": [],
        "profile": {},
        "plan": {},
        "logs": [],
        "memory": {},
        "output": {},
        "pending_questions": user_input.get("pending_questions", []),
    }
    result = app.invoke(state)
    return result.get("output", {})
