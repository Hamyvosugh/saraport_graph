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
)
from app.memory.store import store


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
    """Run the daily coach flow: intake → coach → memory → END"""
    graph = build_graph()
    graph.add_edge("intake", "coach")
    graph.set_entry_point("intake")
    graph.add_edge("coach", "memory")
    graph.add_edge("memory", END)

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