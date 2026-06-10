"""LangGraph nodes for the health coaching workflow."""

import json
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
from app.memory.store import store
from app.graph.state import HealthState


def _llm() -> ChatDeepSeek:
    import os
    # Auto-load from DEEPSEEK_API_KEY in env.local or os.environ
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env.local", override=False)
    load_dotenv(dotenv_path=".env", override=False)
    return ChatDeepSeek(model="deepseek-chat", temperature=0.7)


def _extract_numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text)]


# ---------------------------------------------------------------------------
# Node 1: intake_node
# ---------------------------------------------------------------------------
def intake_node(state: HealthState) -> HealthState:
    """Parse user input and extract structured logs (steps, foods, etc.)."""
    inp = state.get("input", {})
    user_id = state.get("user_id", "unknown")

    detected_logs: dict = {}

    # If a message contains step/food info, extract it
    msg = inp.get("message", "") or inp.get("notes", "")
    if msg:
        nums = _extract_numbers(msg)
        # Heuristic: biggest 3–5 digit number likely steps
        for n in nums:
            if 500 <= n <= 100000:
                detected_logs["steps"] = n
                break
        # Simple food detection via keywords
        foods = re.findall(
            r"(\d+\s*(?:eggs?|pieces?|servings?|cups?|bowls?|plates?|grams?|g|oz|ml|l)[^\.,;]*)",
            msg,
            re.IGNORECASE,
        )
        if foods:
            detected_logs["foods"] = [f.strip() for f in foods]
        elif "egg" in msg.lower():
            # Catch phrases like "3 eggs"
            m = re.search(r"(\d+\s*eggs?)", msg, re.IGNORECASE)
            if m:
                detected_logs["foods"] = [m.group(1).strip()]

    state["detected_logs"] = detected_logs  # type: ignore[index]
    state["messages"] = state.get("messages", []) + [
        {"role": "system", "content": "Intake parsed."}
    ]  # type: ignore[index]
    return state


# ---------------------------------------------------------------------------
# Node 2: planner_node
# ---------------------------------------------------------------------------
def planner_node(state: HealthState) -> HealthState:
    """Create daily targets from user profile."""
    inp = state.get("input", {})
    profile = inp if inp else state.get("profile", {})

    # Simple rule-based targets
    weight = float(profile.get("weight_kg", 75))
    height = float(profile.get("height_cm", 170))
    age = int(profile.get("age", 30))
    gender = profile.get("gender", "male")
    activity = profile.get("activity_level", "moderate")
    goal = profile.get("goal", "maintain")

    # BMR (Mifflin-St Jeor)
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_mult = {"low": 1.2, "moderate": 1.55, "high": 1.9, "very_high": 2.2}
    tdee = bmr * activity_mult.get(activity, 1.55)

    if "lose" in goal.lower() or "reduce" in goal.lower():
        target_cal = int(tdee - 500)
    elif "gain" in goal.lower() or "build" in goal.lower():
        target_cal = int(tdee + 300)
    else:
        target_cal = int(tdee)

    plan = {
        "calories": max(target_cal, 1200),
        "protein_g": int(weight * 1.6),
        "steps": 8000,
        "water_l": 2.5,
        "sleep_h": 7,
    }
    state["plan"] = plan  # type: ignore[index]

    # Persist profile & plan
    user_id = state.get("user_id", "")
    store.set_profile(user_id, profile)
    store.set_plan(user_id, plan)

    state["messages"] = state.get("messages", []) + [
        {"role": "system", "content": f"Plan generated: {plan}"}
    ]  # type: ignore[index]
    return state


# ---------------------------------------------------------------------------
# Node 3: coach_node
# ---------------------------------------------------------------------------
def coach_node(state: HealthState) -> HealthState:
    """Generate coaching reply via DeepSeek."""
    inp = state.get("input", {})
    user_id = state.get("user_id", "")
    plan = state.get("plan", store.get_plan(user_id))
    logs_history = store.get_logs(user_id)
    memory = store.get_memory(user_id)
    detected = state.get("detected_logs", {})  # type: ignore[index]

    msg = inp.get("message", "")
    if not msg:
        state["output"] = {"reply": "No message to respond to."}  # type: ignore[index]
        return state

    context = f"""You are a supportive health & diet coach.
User profile plan: {plan}.
Recent logs: {logs_history[-5:] if logs_history else 'none'}.
User memory summary: {memory.get('summary', 'none')}.
Detected from last message: {detected}.
Reply concisely (2-4 sentences). Acknowledge what the user shared, give brief feedback, and suggest one small next action."""

    try:
        llm = _llm()
        response = llm.invoke([HumanMessage(content=context + f"\n\nUser: {msg}\nCoach:")])
        reply = response.content
    except Exception:
        reply = f"Thanks for sharing! Based on your plan (target {plan.get('calories','?')} cal), keep going. How about a 10-min walk after your next meal?"

    state["output"] = {  # type: ignore[index]
        "reply": reply,
        "detected_logs": detected,
        "next_action": "log tomorrow's progress",
    }
    # Update memory with a brief summary
    mem = store.get_memory(user_id) or {}
    mem["last_coach_reply"] = reply
    store.set_memory(user_id, mem)

    state["messages"] = state.get("messages", []) + [
        {"role": "coach", "content": reply}
    ]  # type: ignore[index]
    return state


# ---------------------------------------------------------------------------
# Node 4: food_vision_node
# ---------------------------------------------------------------------------
def food_vision_node(state: HealthState) -> HealthState:
    """Analyze food from image_url or note (mock fallback for MVP)."""
    inp = state.get("input", {})
    note = inp.get("note", "")
    image_url = inp.get("image_url", "")

    if image_url:
        # If we had vision, we'd call it here. For MVP, use note-based mock.
        pass

    # Simple mock analysis based on note keywords
    note_lower = note.lower() if note else ""
    foods = []
    est_cal = 0
    est_protein = 0
    est_carbs = 0
    est_fat = 0

    food_map = {
        "chicken": ("chicken", 300, 40, 0, 10),
        "rice": ("rice", 200, 4, 45, 1),
        "egg": ("eggs", 140, 12, 1, 9),
        "salad": ("salad", 100, 3, 10, 5),
        "pasta": ("pasta", 350, 12, 55, 8),
        "fish": ("fish", 250, 35, 0, 12),
        "bread": ("bread", 150, 5, 28, 2),
        "fruit": ("fruit", 90, 1, 22, 0),
        "vegetable": ("vegetables", 50, 2, 8, 0),
        "pizza": ("pizza", 400, 18, 40, 20),
        "burger": ("burger", 550, 30, 40, 30),
        "soup": ("soup", 180, 10, 20, 6),
        "steak": ("steak", 400, 45, 0, 22),
        "tofu": ("tofu", 120, 12, 4, 7),
        "beans": ("beans", 180, 12, 30, 2),
    }

    for keyword, (name, cal, prot, carb, fat) in food_map.items():
        if keyword in note_lower:
            foods.append(name)
            est_cal += cal
            est_protein += prot
            est_carbs += carb
            est_fat += fat

    if not foods:
        foods = ["unknown meal"]
        est_cal = 500
        est_protein = 25
        est_carbs = 50
        est_fat = 20

    output = {
        "foods_detected": foods,
        "estimated_calories": est_cal,
        "protein_g": est_protein,
        "carbs_g": est_carbs,
        "fat_g": est_fat,
        "confidence": "low" if not note else "medium",
        "coach_feedback": f"Detected {', '.join(foods)}. Estimated {est_cal} kcal. "
        f"Protein: {est_protein}g. Consider adding vegetables if missing.",
    }

    state["output"] = output  # type: ignore[index]
    return state


# ---------------------------------------------------------------------------
# Node 5: progress_node
# ---------------------------------------------------------------------------
def progress_node(state: HealthState) -> HealthState:
    """Calculate progress score and provide feedback."""
    inp = state.get("input", {})
    user_id = state.get("user_id", "")
    plan = state.get("plan", store.get_plan(user_id))

    # Save the log
    log_entry = {k: v for k, v in inp.items() if k != "user_id"}
    log_entry["timestamp"] = "now"  # simplified
    store.add_log(user_id, log_entry)

    # Score progress
    score = 0
    total = 0
    if "weight_kg" in log_entry and plan.get("calories"):
        # Just tracking = partial progress
        score += 30
        total += 30
    if "steps" in log_entry:
        target = plan.get("steps", 8000)
        achieved = log_entry.get("steps", 0)
        pts = min(40, int(40 * min(achieved / target, 1.5)))
        score += pts
        total += 40
    if "water_l" in log_entry:
        target = plan.get("water_l", 2.5)
        achieved = log_entry.get("water_l", 0)
        pts = min(15, int(15 * min(achieved / target, 1.5)))
        score += pts
        total += 15
    if "sleep_h" in log_entry:
        target = plan.get("sleep_h", 7)
        achieved = log_entry.get("sleep_h", 0)
        pts = min(15, int(15 * min(achieved / target, 1.5)))
        score += pts
        total += 15

    progress_score = int((score / max(total, 1)) * 100) if total else 50

    feedback = ""
    if progress_score >= 80:
        feedback = "Great job! You're on track."
    elif progress_score >= 50:
        feedback = "Good effort. Keep pushing."
    else:
        feedback = "Don't give up. Small consistent steps win."

    state["output"] = {  # type: ignore[index]
        "status": "saved",
        "feedback": feedback,
        "progress_score": progress_score,
    }
    return state


# ---------------------------------------------------------------------------
# Node 6: memory_node
# ---------------------------------------------------------------------------
def memory_node(state: HealthState) -> HealthState:
    """Read/write user memory — called to persist conversation summaries."""
    user_id = state.get("user_id", "")
    inp = state.get("input", {})
    mem = store.get_memory(user_id) or {}

    # Simple: accumulate coach replies as memory summary
    messages = state.get("messages", [])
    if messages:
        coach_msgs = [m["content"] for m in messages if m.get("role") == "coach"]
        mem["summary"] = coach_msgs[-1] if coach_msgs else mem.get("summary", "")
        mem["total_interactions"] = mem.get("total_interactions", 0) + 1

    # Merge any extra memory fields from input
    if "memory_update" in inp:
        mem.update(inp["memory_update"])

    store.set_memory(user_id, mem)
    state["memory"] = mem  # type: ignore[index]
    return state