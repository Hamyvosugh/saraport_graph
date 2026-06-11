"""FastAPI app — Saraport Graph MVP backend."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

# Load env vars from .env.local (or .env) before anything else
load_dotenv(dotenv_path=".env.local", override=False)
load_dotenv(dotenv_path=".env", override=False)

from app.graph.workflow import (
    run_onboarding,
    run_chat,
    run_coach,
    run_daily_log,
    run_food_analysis,
    run_food_breakdown,
    run_food_breakdown_resolve,
)
from app.memory.supabase_store import store

app = FastAPI(
    title="Saraport Graph API",
    description="Health & diet coaching agent backend",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS — allow local Next.js and Vercel previews 11 for frontend development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://saraport.vercel.app",
        "https://saraport-git-main-emoviral.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class OnboardingRequest(BaseModel):
    user_id: str = "demo-user"
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    waist_cm: Optional[float] = None
    goal: str = "maintain"
    activity_level: str = "moderate"
    diet_preferences: str = ""
    limitations: str = ""


class ChatRequest(BaseModel):
    user_id: str = "demo-user"
    message: str


class DailyLogRequest(BaseModel):
    user_id: str = "demo-user"
    weight_kg: Optional[float] = None
    waist_cm: Optional[float] = None
    steps: Optional[int] = None
    water_l: Optional[float] = None
    sleep_h: Optional[float] = None
    notes: str = ""


class FoodAnalyzeRequest(BaseModel):
    user_id: str = "demo-user"
    image_url: Optional[str] = None
    note: str = ""


class FoodBreakdownRequest(BaseModel):
    user_id: str = "demo-user"
    text: str

class FoodLogSaveRequest(BaseModel):
    user_id: str
    foods: list[dict]
    drinks: list[dict] = []

class NutritionTrendsRequest(BaseModel):
    user_id: str
    days: int = 7

class CoachRequest(BaseModel):
    user_id: str
    message: str


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "service": "saraport-graph"}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------
@app.post("/agent/onboarding")
async def onboarding(req: OnboardingRequest):
    """User onboarding: profile → plan → daily targets."""
    result = run_onboarding(req.model_dump())
    return result


@app.post("/agent/chat")
async def chat(req: ChatRequest):
    """Daily coach chat: message → intake → coach reply."""
    result = run_chat(req.model_dump())
    return result


@app.post("/agent/coach")
async def coach(req: CoachRequest):
    """Main coach agent: full tool-calling agent with Supabase persistence."""
    result = run_coach(req.model_dump())
    return result


@app.post("/agent/daily-log")
async def daily_log(req: DailyLogRequest):
    """Manual daily log: save progress and get feedback."""
    result = run_daily_log(req.model_dump())
    return result


@app.post("/agent/food-analyze")
async def food_analyze(req: FoodAnalyzeRequest):
    """Analyze food image or text note (mock vision for MVP)."""
    result = run_food_analysis(req.model_dump())
    return result


class FoodBreakdownResolveRequest(BaseModel):
    user_id: str = "demo-user"
    text: str = ""
    choices: list[dict] = []
    pending_questions: list[dict] = []

@app.post("/agent/food-breakdown")
async def food_breakdown(req: FoodBreakdownRequest):
    """AI food breakdown: analyze free text, search USDA foods, auto-save to DB.
    
    Input: {"user_id": "...", "text": "ناهار ماهی و سالاد خوردم"}
    Returns: auto-saves foods to food_logs, drinks to water_logs.
    If ambiguity: returns pending_questions for user to choose.
    """
    try:
        result = run_food_breakdown(req.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/agent/food-breakdown-resolve")
async def food_breakdown_resolve(req: FoodBreakdownResolveRequest):
    """Resolve pending food ambiguities based on user's choices.
    
    Input: {"user_id": "...", "choices": [{"food_index": 0, "selected_food_id": "uuid"}], "pending_questions": [...]}
    Saves the selected foods to food_logs.
    """
    try:
        result = run_food_breakdown_resolve(req.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/agent/food-log-save")
async def food_log_save(req: FoodLogSaveRequest):
    """Save analyzed foods as individual food_log entries.
    
    Each food item becomes a separate row in food_logs.
    Drinks are saved to water_logs.
    """
    saved_foods = []
    saved_drinks = []
    
    try:
        for food in req.foods:
            entry = store.add_food_log(req.user_id, food)
            saved_foods.append(entry)
        
        for drink in req.drinks:
            entry = store.add_water_log(
                req.user_id,
                amount_ml=drink.get("amount_ml", 250),
                source="ai",
                note=drink.get("name", "")
            )
            saved_drinks.append(entry)
        
        return {
            "success": True,
            "saved_foods": saved_foods,
            "saved_drinks": saved_drinks,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/agent/nutrition-trends/{user_id}")
async def nutrition_trends(user_id: str, days: int = 7):
    """Get aggregated nutrition data for the last N days."""
    from datetime import date, timedelta
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=days)).isoformat()
    
    try:
        aggregation = store.aggregate_food_logs(user_id, from_date, to_date)
        water_today = store.get_water_today(user_id)
        
        return {
            "success": True,
            "data": {
                **aggregation,
                "water_today_ml": water_today,
                "from_date": from_date,
                "to_date": to_date,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/agent/food-logs-today/{user_id}")
async def food_logs_today(user_id: str):
    """Get today's food logs and daily summary for the billboard."""
    from datetime import date
    today = date.today().isoformat()
    
    try:
        logs = store.get_food_logs(user_id, log_date=today)
        water_today = store.get_water_today(user_id)
        
        # Calculate daily totals
        totals = {
            "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0,
            "fiber_g": 0, "sugar_g": 0, "sodium_mg": 0, "potassium_mg": 0,
        }
        for log in logs:
            for key in totals:
                totals[key] += log.get(key, 0) or 0
        
        return {
            "success": True,
            "data": {
                "logs": logs,
                "totals": totals,
                "water_ml": water_today,
                "date": today,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/agent/stats/{user_id}")
async def stats(user_id: str):
    """Return user stats, logs, and insights."""
    profile = store.get_profile(user_id)
    plans = store.get_plan(user_id)
    logs_list = store.get_logs(user_id)
    memory = store.get_memory(user_id)

    latest = logs_list[-1] if logs_list else {}

    # Simple insights from data
    insights = []
    if logs_list:
        weights = [l.get("weight_kg") for l in logs_list if l.get("weight_kg")]
        if len(weights) >= 2:
            delta = weights[-1] - weights[0]
            direction = "down" if delta < 0 else "up"
            insights.append(f"Weight trend: {direction} by {abs(delta):.1f} kg")

        steps_list = [l.get("steps") for l in logs_list if l.get("steps")]
        if steps_list:
            avg_steps = sum(steps_list) / len(steps_list)
            insights.append(f"Average steps: {int(avg_steps)}/day")

    if plans:
        insights.append(f"Daily calorie target: {plans.get('calories', '?')} kcal")

    if memory.get("summary"):
        insights.append(f"Last coach note: {memory['summary'][:100]}")

    return {
        "user_id": user_id,
        "profile": profile,
        "plan": plans,
        "latest": latest,
        "logs": logs_list,
        "insights": insights,
    }