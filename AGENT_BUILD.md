# Saraport Graph — 2 Hour MVP Build Brief

## Goal

Build a separate Python FastAPI + LangGraph backend inside `saraport_graph`.

This backend will power a health/diet coaching web app. The Next.js frontend already exists in the parent project. This backend must expose HTTP APIs that the Next.js app can call.

The goal is not to build a perfect product. The goal is to build a working MVP flow in 2 hours:

User onboarding → health goal planning → daily coaching → food photo analysis → progress tracking → memory.

---

## Tech Stack

* Python
* FastAPI
* LangGraph
* LangChain DEEPSEEK
* DEEPSEEK API
* Supabase/Postgres later
* Railway deployment ready
* `.env` based config

---

## Current Folder

You are working only inside:

```txt
saraport_graph/
```

Do not modify the parent Next.js app yet.

---

## Required Backend Features

Create a FastAPI app with these endpoints:

### 1. Health Check

```txt
GET /
GET /health
```

Return simple JSON.

---

### 2. Onboarding Agent

```txt
POST /agent/onboarding
```

Input:

```json
{
  "user_id": "demo-user",
  "age": 43,
  "gender": "male",
  "height_cm": 183,
  "weight_kg": 87,
  "waist_cm": 98,
  "goal": "lose weight and reduce waist",
  "activity_level": "low",
  "diet_preferences": "high protein, simple meals",
  "limitations": "none"
}
```

Output:

```json
{
  "user_id": "demo-user",
  "summary": "...",
  "initial_plan": "...",
  "daily_targets": {
    "calories": 1800,
    "protein_g": 140,
    "steps": 8000,
    "water_l": 2.5,
    "sleep_h": 7
  }
}
```

---

### 3. Daily Coach Agent

```txt
POST /agent/chat
```

Input:

```json
{
  "user_id": "demo-user",
  "message": "Today I ate 3 eggs and walked 4000 steps."
}
```

Output:

```json
{
  "reply": "...",
  "detected_logs": {
    "steps": 4000,
    "foods": ["3 eggs"]
  },
  "next_action": "..."
}
```

---

### 4. Manual Daily Log

```txt
POST /agent/daily-log
```

Input:

```json
{
  "user_id": "demo-user",
  "weight_kg": 86.5,
  "waist_cm": 97,
  "steps": 6500,
  "water_l": 2.2,
  "sleep_h": 6.5,
  "notes": "felt hungry at night"
}
```

Output:

```json
{
  "status": "saved",
  "feedback": "...",
  "progress_score": 72
}
```

For MVP, in-memory storage is acceptable.

---

### 5. Food Image Analysis Agent

```txt
POST /agent/food-analyze
```

For MVP accept either:

```json
{
  "user_id": "demo-user",
  "image_url": "https://example.com/food.jpg",
  "note": "This is my lunch"
}
```

or simple base64 image field if easier.

Output:

```json
{
  "foods_detected": ["..."],
  "estimated_calories": 500,
  "protein_g": 35,
  "carbs_g": 45,
  "fat_g": 18,
  "confidence": "medium",
  "coach_feedback": "..."
}
```

Use DEEPSEEK vision-capable model if possible. If image handling is not ready, create a mock fallback that works with `note`.

---

### 6. Stats Endpoint

```txt
GET /agent/stats/{user_id}
```

Output:

```json
{
  "user_id": "demo-user",
  "latest": {},
  "logs": [],
  "insights": ["..."]
}
```

---

## LangGraph Architecture

Create a simple graph with these nodes:

1. `intake_node`
2. `planner_node`
3. `coach_node`
4. `food_vision_node`
5. `progress_node`
6. `memory_node`

For MVP, it is acceptable to use one graph file and conditionally call functions per endpoint.

Use typed state.

Example state fields:

```python
class HealthState(TypedDict):
    user_id: str
    input: dict
    messages: list
    profile: dict
    plan: dict
    logs: list
    memory: dict
    output: dict
```

---

## Files To Create

Create this structure:

```txt
saraport_graph/
  app/
    __init__.py
    main.py
    graph/
      __init__.py
      state.py
      nodes.py
      workflow.py
    memory/
      __init__.py
      store.py
  .env.example
  requirements.txt
  Procfile
  railway.json
  README.md
```

---

## Implementation Rules

* Keep code simple.
* Do not over-engineer.
* No database integration in first pass unless very fast.
* Use in-memory storage first.
* Make Railway deployment ready.
* Use Pydantic request/response models.
* Add CORS for local Next.js and Vercel.
* Use environment variable:

```txt
DEEPSEEK_API_KEY=
```

* Use model:

```txt
gpt-4.1-mini
```

or another available DEEPSEEK chat model.

---

## Success Criteria

After implementation, these commands must work:

```bash
uvicorn app.main:app --reload --port 8000
```

Then:

```txt
http://localhost:8000/health
```

must return JSON.

The API docs must work:

```txt
http://localhost:8000/docs
```

---

## Final Output Expected From Cline

After building, report:

1. Files created
2. How to run locally
3. Required env variables
4. Test endpoints
5. Any missing items

Do not ask unnecessary questions. Build the MVP directly.
