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
# ---------------------------------------------------------------------------
# Node 6: food_breakdown_node — auto-save foods, return summary or questions
# ---------------------------------------------------------------------------
def food_breakdown_node(state: HealthState) -> HealthState:
    """Parse free-text food description, search USDA, save automatically.
    
    If ambiguity (multiple USDA matches), returns pending_questions.
    Otherwise saves all foods to food_logs and drinks to water_logs."""
    inp = state.get("input", {})
    text = inp.get("text", "") or inp.get("note", "")
    user_id = state.get("user_id", "")

    if not text:
        state["output"] = {"error": "هیچ توضیحی برای تحلیل وجود ندارد"}
        return state

    from app.memory.supabase_store import store as db

    llm = _llm()
    
    # Step 1: LLM extracts foods/drinks with better Persian awareness
    extraction_prompt = f"""You are a precise food-entity extractor for a Persian/Iranian user.
Given a free-text description (usually in Persian/Farsi) of what they ate/drank, extract EVERY individual food and drink.

CRITICAL RULES:
1. Extract each food/drink item separately
2. For each food, provide "name_fa" (Persian name) and "name_en" (English for database search)
3. Estimate grams for each food (1 serving ≈ 300-400g, 1 cup ≈ 200-250g, 1 piece ≈ 100-150g)
4. Determine meal_type: "breakfast", "lunch", "dinner", or "snack"
5. For traditional Iranian dishes (قرمه سبزی, فسنجان, کباب, etc.) set "needs_decomposition": true
6. Put drinks (water, tea, coffee, juice, milk, دوغ, etc.) in "drinks" array with amount_ml

User text: "{text}"

Output ONLY valid JSON:
{{
  "foods": [
    {{"name_fa": "ماهی", "name_en": "fish", "meal_type": "lunch", "estimated_grams": 300, "needs_decomposition": false}}
  ],
  "drinks": [
    {{"name_fa": "آب", "name_en": "water", "amount_ml": 300}}
  ]
}}"""

    extracted_items = {"foods": [], "drinks": []}
    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        content = response.content.strip()
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            extracted_items = json.loads(json_match.group(0))
    except Exception as e:
        state["output"] = {"error": f"خطا در استخراج غذاها: {str(e)}"}
        return state

    pending_questions = []
    saved_foods = []
    saved_drinks = []
    now = __import__("datetime").datetime.now().isoformat()
    log_date = inp.get("log_date", __import__("datetime").date.today().isoformat())

    # Helper: build a full DB entry from a USDA match
    def build_food_entry(match: dict, name_fa: str, meal_type: str, grams: float, source: str, confidence: float) -> dict:
        scale = grams / 100.0
        return {
            "food_name": name_fa,
            "food_id": match.get("id"),
            "meal_type": meal_type,
            "quantity": 1,
            "unit": "serving",
            "serving_grams": round(grams),
            "calories": int(round((match.get("calories") or 0) * scale)),
            "protein_g": round((match.get("protein_g") or 0) * scale, 1),
            "carbs_g": round((match.get("carbs_g") or 0) * scale, 1),
            "fat_g": round((match.get("fat_g") or 0) * scale, 1),
            "fiber_g": round((match.get("fiber_g") or 0) * scale, 1),
            "sugar_g": round((match.get("sugar_g") or 0) * scale, 1),
            "sodium_mg": round((match.get("sodium_mg") or 0) * scale, 1),
            "potassium_mg": round((match.get("potassium_mg") or 0) * scale, 1),
            "calcium_mg": round((match.get("calcium_mg") or 0) * scale, 1),
            "iron_mg": round((match.get("iron_mg") or 0) * scale, 1),
            "magnesium_mg": round((match.get("magnesium_mg") or 0) * scale, 1) if match.get("magnesium_mg") else 0,
            "phosphorus_mg": round((match.get("phosphorus_mg") or 0) * scale, 1) if match.get("phosphorus_mg") else 0,
            "zinc_mg": round((match.get("zinc_mg") or 0) * scale, 1) if match.get("zinc_mg") else 0,
            "selenium_mcg": round((match.get("selenium_mcg") or 0) * scale, 1) if match.get("selenium_mcg") else 0,
            "cholesterol_mg": round((match.get("cholesterol_mg") or 0) * scale, 1) if match.get("cholesterol_mg") else 0,
            "saturated_fat_g": round((match.get("saturated_fat_g") or 0) * scale, 1) if match.get("saturated_fat_g") else 0,
            "monounsaturated_fat_g": round((match.get("monounsaturated_fat_g") or 0) * scale, 1) if match.get("monounsaturated_fat_g") else 0,
            "polyunsaturated_fat_g": round((match.get("polyunsaturated_fat_g") or 0) * scale, 1) if match.get("polyunsaturated_fat_g") else 0,
            "vitamin_a_mcg": round((match.get("vitamin_a_mcg") or 0) * scale, 1) if match.get("vitamin_a_mcg") else 0,
            "vitamin_c_mg": round((match.get("vitamin_c_mg") or 0) * scale, 1) if match.get("vitamin_c_mg") else 0,
            "vitamin_d_mcg": round((match.get("vitamin_d_mcg") or 0) * scale, 1) if match.get("vitamin_d_mcg") else 0,
            "vitamin_e_mg": round((match.get("vitamin_e_mg") or 0) * scale, 1) if match.get("vitamin_e_mg") else 0,
            "vitamin_k_mcg": round((match.get("vitamin_k_mcg") or 0) * scale, 1) if match.get("vitamin_k_mcg") else 0,
            "vitamin_b1_mg": round((match.get("vitamin_b1_mg") or 0) * scale, 1) if match.get("vitamin_b1_mg") else 0,
            "vitamin_b2_mg": round((match.get("vitamin_b2_mg") or 0) * scale, 1) if match.get("vitamin_b2_mg") else 0,
            "vitamin_b3_mg": round((match.get("vitamin_b3_mg") or 0) * scale, 1) if match.get("vitamin_b3_mg") else 0,
            "vitamin_b6_mg": round((match.get("vitamin_b6_mg") or 0) * scale, 1) if match.get("vitamin_b6_mg") else 0,
            "vitamin_b12_mcg": round((match.get("vitamin_b12_mcg") or 0) * scale, 1) if match.get("vitamin_b12_mcg") else 0,
            "source": source,
            "confidence": confidence,
            "log_date": log_date,
            "logged_at": now,
            "note": text,
        }

    # Check if the user specified quantities/amounts
    # If the text is vague, ask the LLM if we need to ask for more details
    vagueness_prompt = f"""Analyze this food description. Does the user clearly state WHAT they ate and HOW MUCH?
If the food name is clear but quantity is vague (e.g. "قیمه خوردم" without grams), flag it as needing quantity.
If the food type is vague (e.g. "میوه خوردم" without saying which fruit), flag it as needing specification.

User text: "{text}"
Extracted items: {json.dumps(extracted_items, ensure_ascii=False)}

Output ONLY valid JSON:
{{
  "needs_quantity_for": [],  // list of food indices (0-based) that need quantity asked
  "needs_specification_for": [],  // list of food indices that need clarification (what exactly?)
  "suggestion": ""  // a helpful Persian question to ask the user
}}"""

    clarification_needed = {"needs_quantity_for": [], "needs_specification_for": []}
    try:
        vr = llm.invoke([HumanMessage(content=vagueness_prompt)])
        vc = vr.content.strip()
        jm = re.search(r'\{.*\}', vc, re.DOTALL)
        if jm:
            clarification_needed = json.loads(jm.group(0))
    except:
        pass

    # If the user hasn't specified quantities/types, return questions first
    all_vague = (clarification_needed.get("needs_quantity_for", []) + 
                 clarification_needed.get("needs_specification_for", []))
    if all_vague:
        pending_questions.append({
            "question": clarification_needed.get("suggestion", "لطفاً اطلاعات بیشتری درباره غذای خود بدهید:"),
            "type": "clarification",
            "food_index": 0,
            "original_text": text,
        })
        state["output"] = {
            "status": "questions_pending",
            "saved_so_far": [],
            "saved_drinks": [],
            "pending_questions": pending_questions,
            "message": clarification_needed.get("suggestion", "لطفاً دقیق‌تر توضیح دهید چه خوردید و چقدر."),
        }
        state["pending_questions"] = pending_questions
        return state

    # Process each food
    for food_item in extracted_items.get("foods", []):
        name_fa = food_item.get("name_fa", food_item.get("name", ""))
        name_en = food_item.get("name_en", food_item.get("name", ""))
        est_grams = food_item.get("estimated_grams", 200)
        meal_type = food_item.get("meal_type", "snack")
        needs_decomp = food_item.get("needs_decomposition", False)

        # Skip items with empty names
        if not name_fa:
            continue

        # Search USDA
        matched_foods = db.search_foods(name_en, limit=4)
        
        if len(matched_foods) >= 2:
            # Ambiguity: ask user to choose (with "none" option)
            choices = []
            for m in matched_foods[:3]:
                choices.append({
                    "food_id": m.get("id"),
                    "description": m.get("description", ""),
                    "calories_per_100g": m.get("calories", 0),
                    "category": m.get("food_category", ""),
                })
            # Add "none of these" and "cancel" options
            choices.append({
                "food_id": "__none__",
                "description": "هیچکدام - رد کردن این مورد",
                "calories_per_100g": 0,
                "category": "skip",
            })
            choices.append({
                "food_id": "__cancel__",
                "description": "انصراف - لغو کل تحلیل",
                "calories_per_100g": 0,
                "category": "cancel",
            })
            pending_questions.append({
                "question": f"برای «{name_fa}» چند گزینه پیدا شد. کدام را انتخاب می‌کنید؟",
                "type": "food_choice",
                "food_index": len(pending_questions),
                "original_name_fa": name_fa,
                "original_name_en": name_en,
                "meal_type": meal_type,
                "estimated_grams": est_grams,
                "choices": choices,
            })
        elif len(matched_foods) == 1:
            # Single match → save directly
            entry = build_food_entry(matched_foods[0], name_fa, meal_type, est_grams, "usda_foundation", 0.9)
            saved = db.add_food_log(user_id, entry)
            saved_foods.append(saved)
        elif needs_decomp:
            # Decompose Iranian dish
            decomp_prompt = f"""Decompose the Persian dish "{name_fa}" into basic ingredients.
Estimate weight percentage of each ingredient.
Output ONLY valid JSON array:
[{{"name_en": "beef", "name_fa": "گوشت", "pct": 30}}, {{"name_en": "rice", "name_fa": "برنج", "pct": 40}}]"""
            try:
                decomp_resp = llm.invoke([HumanMessage(content=decomp_prompt)])
                dc = decomp_resp.content.strip()
                jm = re.search(r'\[.*\]', dc, re.DOTALL)
                if jm:
                    ingredients = json.loads(jm.group(0))
                    for ing in ingredients:
                        ing_name_en = ing.get("name_en", "")
                        ing_name_fa = ing.get("name_fa", ing_name_en)
                        ing_pct = ing.get("pct", 100) / 100.0
                        ing_grams = round(est_grams * ing_pct)
                        if not ing_name_fa and not ing_name_en:
                            continue
                        
                        ing_matches = db.search_foods(ing_name_en, limit=2)
                        if ing_matches:
                            entry = build_food_entry(ing_matches[0], f"{name_fa} - {ing_name_fa}", meal_type, ing_grams, "ai_decomposed", 0.6)
                            saved = db.add_food_log(user_id, entry)
                            saved_foods.append(saved)
                        else:
                            try:
                                est_p = f"""Estimate nutritional values for {ing_grams}g of "{ing_name_en}".
Output ONLY JSON: {{"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}}"""
                                er = llm.invoke([HumanMessage(content=est_p)])
                                e = json.loads(re.search(r'\{.*\}', er.content.strip(), re.DOTALL).group(0)) if re.search(r'\{.*\}', er.content.strip(), re.DOTALL) else {}
                                db.add_food_log(user_id, {
                                    "food_name": f"{name_fa} - {ing_name_fa}",
                                    "meal_type": meal_type, "quantity": round(ing_pct, 2),
                                    "serving_grams": ing_grams, "unit": "portion",
                                    "calories": e.get("calories", 100),
                                    "protein_g": e.get("protein_g", 3),
                                    "carbs_g": e.get("carbs_g", 10),
                                    "fat_g": e.get("fat_g", 5),
                                    "source": "ai_estimated", "confidence": 0.4,
                                    "log_date": log_date, "logged_at": now,
                                })
                                saved_foods.append({"food_name": f"{name_fa} - {ing_name_fa}", "calories": e.get("calories", 100)})
                            except:
                                pass
            except:
                try:
                    ep = f"""Estimate nutritional values for {est_grams}g of Persian dish "{name_fa}".
Output ONLY JSON: {{"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}}"""
                    er = llm.invoke([HumanMessage(content=ep)])
                    e = json.loads(re.search(r'\{.*\}', er.content.strip(), re.DOTALL).group(0)) if re.search(r'\{.*\}', er.content.strip(), re.DOTALL) else {}
                    db.add_food_log(user_id, {
                        "food_name": name_fa, "meal_type": meal_type,
                        "serving_grams": est_grams, "unit": "serving",
                        "calories": e.get("calories", 400),
                        "protein_g": e.get("protein_g", 15),
                        "carbs_g": e.get("carbs_g", 40),
                        "fat_g": e.get("fat_g", 18),
                        "source": "ai_estimated", "confidence": 0.35,
                        "log_date": log_date, "logged_at": now,
                    })
                    saved_foods.append({"food_name": name_fa, "calories": e.get("calories", 400)})
                except:
                    pass
        else:
            try:
                ep = f"""Estimate nutritional values for {est_grams}g of "{name_fa}" (English: {name_en}).
Output ONLY JSON: {{"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "fiber_g": 0, "sugar_g": 0}}"""
                er = llm.invoke([HumanMessage(content=ep)])
                e = json.loads(re.search(r'\{.*\}', er.content.strip(), re.DOTALL).group(0)) if re.search(r'\{.*\}', er.content.strip(), re.DOTALL) else {}
                db.add_food_log(user_id, {
                    "food_name": name_fa, "meal_type": meal_type,
                    "serving_grams": est_grams, "unit": "serving",
                    "calories": e.get("calories", 200),
                    "protein_g": e.get("protein_g", 5),
                    "carbs_g": e.get("carbs_g", 20),
                    "fat_g": e.get("fat_g", 10),
                    "fiber_g": e.get("fiber_g", 1),
                    "sugar_g": e.get("sugar_g", 5),
                    "source": "ai_estimated", "confidence": 0.5,
                    "log_date": log_date, "logged_at": now,
                })
                saved_foods.append({"food_name": name_fa, "calories": e.get("calories", 200)})
            except:
                db.add_food_log(user_id, {
                    "food_name": name_fa, "meal_type": meal_type,
                    "serving_grams": est_grams, "unit": "serving",
                    "calories": 300, "source": "ai_estimated", "confidence": 0.3,
                    "log_date": log_date, "logged_at": now,
                })
                saved_foods.append({"food_name": name_fa, "calories": 300})

    # Save drinks to water_logs
    for drink in extracted_items.get("drinks", []):
        drink_name = drink.get("name_fa", drink.get("name", ""))
        amount_ml = drink.get("amount_ml", 250)
        db.add_water_log(user_id, amount_ml, source="ai", note=drink_name)
        saved_drinks.append({"name": drink_name, "amount_ml": amount_ml})

    # If there are pending questions, return them (don't save those yet)
    if pending_questions:
        state["pending_questions"] = pending_questions
        saved_preview = []
        for sf in saved_foods:
            saved_preview.append({
                "food_name": sf.get("food_name", ""),
                "calories": sf.get("calories", 0),
                "serving_grams": sf.get("serving_grams", 0),
            })
        state["output"] = {
            "status": "questions_pending",
            "saved_so_far": saved_preview,
            "saved_drinks": saved_drinks,
            "pending_questions": pending_questions,
            "message": f"{len(saved_foods)} ماده غذایی مستقیماً ذخیره شد. {len(pending_questions)} مورد نیاز به انتخاب شما دارد.",
        }
        return state

    # All resolved — return success summary
    saved_preview = []
    total_cal = 0
    for sf in saved_foods:
        cal = sf.get("calories", 0) or 0
        saved_preview.append({
            "food_name": sf.get("food_name", ""),
            "calories": cal,
            "serving_grams": sf.get("serving_grams", 0),
        })
        total_cal += cal

    total_water = sum(d.get("amount_ml", 0) for d in saved_drinks)

    state["output"] = {
        "status": "saved",
        "saved_foods": saved_preview,
        "saved_drinks": saved_drinks,
        "total_foods": len(saved_foods),
        "total_calories": total_cal,
        "total_water_ml": total_water,
        "message": f"✅ {len(saved_foods)} ماده غذایی با مجموع {total_cal} کالری ثبت شد." + (f" {total_water}ml نوشیدنی." if total_water else ""),
    }
    return state


# ---------------------------------------------------------------------------
# Node 6b: food_breakdown_resolve_node — handle user's answer to ambiguity
# ---------------------------------------------------------------------------
def food_breakdown_resolve_node(state: HealthState) -> HealthState:
    """Resolve pending food ambiguities based on user's choices.
    Handles __none__ (skip) and __cancel__ (abort) special values."""
    inp = state.get("input", {})
    user_id = state.get("user_id", "")
    choices = inp.get("choices", [])  # [{food_index: 0, selected_food_id: "uuid"}]
    text = inp.get("text", "")

    # Filter out special choices handled by frontend
    valid_choices = [c for c in choices if c.get("selected_food_id") not in ("__none__", "__cancel__")]
    
    if not valid_choices:
        state["output"] = {
            "status": "saved",
            "saved_foods": [],
            "total_foods": 0,
            "message": "مورد رد شد.",
        }
        return state

    from app.memory.supabase_store import store as db

    now = __import__("datetime").datetime.now().isoformat()
    log_date = inp.get("log_date", __import__("datetime").date.today().isoformat())

    saved_foods = []

    for choice in valid_choices:
        idx = choice.get("food_index", 0)
        selected_id = choice.get("selected_food_id", "")
        pending = state.get("pending_questions") or []
        q = pending[idx] if idx < len(pending) else {}

        match = db.get_food_by_id(selected_id)
        if match:
            scale = q.get("estimated_grams", 200) / 100.0
            entry = {
                "food_name": q.get("original_name_fa", ""),
                "food_id": match.get("id"),
                "meal_type": q.get("meal_type", "snack"),
                "quantity": 1,
                "unit": "serving",
                "serving_grams": round(q.get("estimated_grams", 200)),
                "calories": int(round((match.get("calories") or 0) * scale)),
                "protein_g": round((match.get("protein_g") or 0) * scale, 1),
                "carbs_g": round((match.get("carbs_g") or 0) * scale, 1),
                "fat_g": round((match.get("fat_g") or 0) * scale, 1),
                "fiber_g": round((match.get("fiber_g") or 0) * scale, 1),
                "sugar_g": round((match.get("sugar_g") or 0) * scale, 1),
                "sodium_mg": round((match.get("sodium_mg") or 0) * scale, 1),
                "potassium_mg": round((match.get("potassium_mg") or 0) * scale, 1),
                "source": "usda_foundation",
                "confidence": 0.85,
                "log_date": log_date,
                "logged_at": now,
            }
            saved = db.add_food_log(user_id, entry)
            saved_foods.append(saved)

    state["pending_questions"] = None
    state["output"] = {
        "status": "saved",
        "saved_foods": [{"food_name": s.get("food_name", ""), "calories": s.get("calories", 0)} for s in saved_foods],
        "total_foods": len(saved_foods),
        "message": f"✅ {len(saved_foods)} ماده غذایی دیگر با انتخاب شما ثبت شد." if saved_foods else "مورد انتخاب نشد.",
    }
    return state


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