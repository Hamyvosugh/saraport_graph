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
    run_daily_log,
    run_food_analysis,
)
from app.memory.store import store

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