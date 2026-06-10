# Saraport Graph

FastAPI + LangGraph health coaching backend.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env.local
# Edit .env.local with your DEEPSEEK_API_KEY
```

## Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Railway Deploy

Push to Railway with the Procfile and railway.json already configured.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/agent/onboarding` | User onboarding — profile & plan |
| POST | `/agent/chat` | Daily coach chat |
| POST | `/agent/daily-log` | Manual daily log & progress |
| POST | `/agent/food-analyze` | Food analysis (image or note) |
| GET | `/agent/stats/{user_id}` | User stats, logs, insights |