"""Supabase-backed store — replaces InMemoryStore for persistence."""

import os
import json
from typing import Any
from datetime import date, datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local", override=False)
load_dotenv(dotenv_path=".env", override=False)


def _get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required")
    return create_client(url, key)


class SupabaseStore:
    """Persistent store backed by Supabase Postgres."""

    def __init__(self):
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = _get_supabase()
        return self._client

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------
    def get_profile(self, user_id: str) -> dict:
        res = self.client.table("profiles").select("*").eq("id", user_id).execute()
        rows = res.data or []
        return rows[0] if rows else {}

    def set_profile(self, user_id: str, profile: dict) -> None:
        existing = self.get_profile(user_id)
        data = {**profile, "id": user_id}
        if "user_id" in data:
            del data["user_id"]
        if existing:
            self.client.table("profiles").update(data).eq("id", user_id).execute()
        else:
            # For MVP, we bypass RLS using service_role
            self.client.table("profiles").insert(
                {"id": user_id, **data}
            ).execute()

    # ------------------------------------------------------------------
    # Plan (stored in goals table, status='active' = current plan)
    # ------------------------------------------------------------------
    def get_plan(self, user_id: str) -> dict:
        """Return current active goal as plan dict."""
        res = (
            self.client.table("goals")
            .select("*")
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if rows:
            row = rows[0]
            return {
                "calories": row.get("target_weight_kg"),  # we store plan here
                "goal_type": row.get("goal_type"),
            }
        return {}

    def set_plan(self, user_id: str, plan: dict) -> None:
        # Upsert: one active goal per user
        existing = self.get_plan(user_id)
        goal_data = {
            "user_id": user_id,
            "goal_type": "health_plan",
            "target_weight_kg": plan.get("calories"),
            "target_waist_cm": plan.get("steps"),
            "status": "active",
        }
        # Store full plan as extended fields
        if existing:
            # update existing
            res = (
                self.client.table("goals")
                .select("id")
                .eq("user_id", user_id)
                .eq("status", "active")
                .execute()
            )
            if res.data:
                goal_id = res.data[0]["id"]
                self.client.table("goals").update(goal_data).eq("id", goal_id).execute()
        else:
            self.client.table("goals").insert(goal_data).execute()

    # ------------------------------------------------------------------
    # Logs (legacy daily_logs table)
    # ------------------------------------------------------------------
    def get_logs(self, user_id: str, limit: int = 50) -> list[dict]:
        res = (
            self.client.table("daily_logs")
            .select("*")
            .eq("user_id", user_id)
            .order("log_date", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []

    def add_log(self, user_id: str, log: dict) -> None:
        self.client.table("daily_logs").insert({
            "user_id": user_id,
            "log_date": date.today().isoformat(),
            **{k: v for k, v in log.items() if k not in ("user_id", "timestamp")},
        }).execute()

    # ------------------------------------------------------------------
    # Log Entries (new flexible log_entries table)
    # ------------------------------------------------------------------
    def get_log_entries(
        self, user_id: str, days: int = 7, category: str = None
    ) -> list[dict]:
        """Fetch log_entries for the last N days, optionally filtered by category."""
        from_date = (date.today().isoformat())
        query = (
            self.client.table("log_entries")
            .select("*")
            .eq("user_id", user_id)
            .order("log_date", desc=True)
            .order("recorded_at", desc=True)
        )
        if category:
            query = query.eq("category", category)
        # We fetch up to 500 rows and filter by date in Python
        # because the gte on log_date is computed relative to today
        res = query.limit(500).execute()
        rows = res.data or []
        # Filter to last N days
        import datetime as dt
        cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        return [r for r in rows if r.get("log_date", "") >= cutoff]

    def get_log_entries_range(
        self, user_id: str, from_date: str, to_date: str
    ) -> list[dict]:
        """Fetch log_entries for a specific date range."""
        res = (
            self.client.table("log_entries")
            .select("*")
            .eq("user_id", user_id)
            .gte("log_date", from_date)
            .lte("log_date", to_date)
            .order("log_date", desc=True)
            .order("recorded_at", desc=True)
            .execute()
        )
        return res.data or []

    # ------------------------------------------------------------------
    # Memory (coach_memory table — simple key-value)
    # ------------------------------------------------------------------
    def get_memory(self, user_id: str) -> dict:
        res = (
            self.client.table("coach_memory")
            .select("*")
            .eq("user_id", user_id)
            .eq("key", "coach_summary")
            .execute()
        )
        rows = res.data or []
        return rows[0].get("value", {}) if rows else {}

    def set_memory(self, user_id: str, memory: dict) -> None:
        existing = (
            self.client.table("coach_memory")
            .select("id")
            .eq("user_id", user_id)
            .eq("key", "coach_summary")
            .execute()
        )
        if existing.data:
            self.client.table("coach_memory").update(
                {"value": json.dumps(memory) if isinstance(memory, dict) else memory}
            ).eq("id", existing.data[0]["id"]).execute()
        else:
            self.client.table("coach_memory").insert({
                "user_id": user_id,
                "key": "coach_summary",
                "value": memory if isinstance(memory, dict) else {},
            }).execute()

    # ------------------------------------------------------------------
    # Coach Messages
    # ------------------------------------------------------------------
    def get_messages(self, user_id: str, limit: int = 50) -> list[dict]:
        res = (
            self.client.table("coach_messages")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return res.data or []

    def add_message(self, user_id: str, role: str, content: str) -> dict:
        res = (
            self.client.table("coach_messages")
            .insert({"user_id": user_id, "role": role, "content": content})
            .execute()
        )
        return res.data[0] if res.data else {}

    # ------------------------------------------------------------------
    # Food Logs
    # ------------------------------------------------------------------
    def add_food_log(self, user_id: str, food_data: dict) -> dict:
        """Insert a single food log entry with all nutritional columns.
        Returns the inserted row."""
        now = datetime.now().isoformat()
        row = {
            "user_id": user_id,
            "log_date": food_data.get("log_date", date.today().isoformat()),
            "food_id": food_data.get("food_id"),
            "food_name": food_data.get("food_name", ""),
            "meal_type": food_data.get("meal_type", "snack"),
            "quantity": food_data.get("quantity", 1),
            "unit": food_data.get("unit", "serving"),
            "serving_grams": food_data.get("serving_grams", 100),
            "calories": food_data.get("calories", 0),
            "protein_g": food_data.get("protein_g", 0),
            "carbs_g": food_data.get("carbs_g", 0),
            "fat_g": food_data.get("fat_g", 0),
            "fiber_g": food_data.get("fiber_g", 0),
            "sugar_g": food_data.get("sugar_g", 0),
            "sodium_mg": food_data.get("sodium_mg", 0),
            "potassium_mg": food_data.get("potassium_mg", 0),
            "calcium_mg": food_data.get("calcium_mg", 0),
            "iron_mg": food_data.get("iron_mg", 0),
            "magnesium_mg": food_data.get("magnesium_mg", 0),
            "phosphorus_mg": food_data.get("phosphorus_mg", 0),
            "zinc_mg": food_data.get("zinc_mg", 0),
            "selenium_mcg": food_data.get("selenium_mcg", 0),
            "cholesterol_mg": food_data.get("cholesterol_mg", 0),
            "saturated_fat_g": food_data.get("saturated_fat_g", 0),
            "monounsaturated_fat_g": food_data.get("monounsaturated_fat_g", 0),
            "polyunsaturated_fat_g": food_data.get("polyunsaturated_fat_g", 0),
            "vitamin_a_mcg": food_data.get("vitamin_a_mcg", 0),
            "vitamin_c_mg": food_data.get("vitamin_c_mg", 0),
            "vitamin_d_mcg": food_data.get("vitamin_d_mcg", 0),
            "vitamin_e_mg": food_data.get("vitamin_e_mg", 0),
            "vitamin_k_mcg": food_data.get("vitamin_k_mcg", 0),
            "vitamin_b1_mg": food_data.get("vitamin_b1_mg", 0),
            "vitamin_b2_mg": food_data.get("vitamin_b2_mg", 0),
            "vitamin_b3_mg": food_data.get("vitamin_b3_mg", 0),
            "vitamin_b6_mg": food_data.get("vitamin_b6_mg", 0),
            "vitamin_b12_mcg": food_data.get("vitamin_b12_mcg", 0),
            "source": food_data.get("source", "ai"),
            "brand_name": food_data.get("brand_name"),
            "barcode": food_data.get("barcode"),
            "confidence": food_data.get("confidence", 0.7),
            "ai_model": food_data.get("ai_model", "deepseek"),
            "ai_raw_response": food_data.get("ai_raw_response"),
            "note": food_data.get("note", ""),
            "logged_at": food_data.get("logged_at", now),
        }
        # Remove None values for optional fields
        row = {k: v for k, v in row.items() if v is not None}
        res = self.client.table("food_logs").insert(row).execute()
        return res.data[0] if res.data else {}

    def get_food_logs(
        self, user_id: str, log_date: str = None, limit: int = 100
    ) -> list[dict]:
        """Fetch food logs, optionally filtered by date."""
        query = (
            self.client.table("food_logs")
            .select("*")
            .eq("user_id", user_id)
            .order("logged_at", desc=True)
            .limit(limit)
        )
        if log_date:
            query = query.eq("log_date", log_date)
        res = query.execute()
        return res.data or []

    def get_food_logs_range(
        self, user_id: str, from_date: str, to_date: str
    ) -> list[dict]:
        """Fetch food logs for a date range."""
        res = (
            self.client.table("food_logs")
            .select("*")
            .eq("user_id", user_id)
            .gte("log_date", from_date)
            .lte("log_date", to_date)
            .order("log_date", desc=True)
            .execute()
        )
        return res.data or []

    def update_food_log(self, log_id: str, user_id: str, updates: dict) -> dict:
        """Update a food log entry."""
        res = (
            self.client.table("food_logs")
            .update(updates)
            .eq("id", log_id)
            .eq("user_id", user_id)
            .execute()
        )
        return res.data[0] if res.data else {}

    def delete_food_log(self, log_id: str, user_id: str) -> bool:
        """Delete a food log entry."""
        res = (
            self.client.table("food_logs")
            .delete()
            .eq("id", log_id)
            .eq("user_id", user_id)
            .execute()
        )
        return bool(res.data)

    # ------------------------------------------------------------------
    # Food Search (USDA foods table)
    # ------------------------------------------------------------------
    def search_foods(self, query: str, limit: int = 5) -> list[dict]:
        """Search the foods table by description (English)."""
        res = (
            self.client.table("foods")
            .select("*")
            .ilike("description", f"%{query}%")
            .limit(limit)
            .execute()
        )
        return res.data or []

    def get_food_by_id(self, food_id: str) -> dict:
        """Get a single food by its UUID."""
        res = (
            self.client.table("foods")
            .select("*")
            .eq("id", food_id)
            .execute()
        )
        return res.data[0] if res.data else {}

    # ------------------------------------------------------------------
    # Water Logs
    # ------------------------------------------------------------------
    def add_water_log(self, user_id: str, amount_ml: int, source: str = "manual", note: str = None) -> dict:
        """Insert a water consumption log."""
        row = {
            "user_id": user_id,
            "amount_ml": amount_ml,
            "source": source,
            "note": note,
        }
        row = {k: v for k, v in row.items() if v is not None}
        res = self.client.table("water_logs").insert(row).execute()
        return res.data[0] if res.data else {}

    def get_water_logs(self, user_id: str, from_date: str = None, to_date: str = None, limit: int = 50) -> list[dict]:
        """Fetch water logs, optionally filtered by date range on logged_at."""
        query = (
            self.client.table("water_logs")
            .select("*")
            .eq("user_id", user_id)
            .order("logged_at", desc=True)
            .limit(limit)
        )
        if from_date:
            query = query.gte("logged_at", f"{from_date}T00:00:00")
        if to_date:
            query = query.lte("logged_at", f"{to_date}T23:59:59")
        res = query.execute()
        return res.data or []

    def get_water_today(self, user_id: str) -> int:
        """Get total water intake for today in ml."""
        today = date.today().isoformat()
        logs = self.get_water_logs(user_id, from_date=today, to_date=today)
        return sum(l.get("amount_ml", 0) for l in logs)

    # ------------------------------------------------------------------
    # Food Logs Aggregation for Nutrition Trends
    # ------------------------------------------------------------------
    def aggregate_food_logs(
        self, user_id: str, from_date: str, to_date: str
    ) -> dict:
        """Aggregate nutritional values for a date range."""
        logs = self.get_food_logs_range(user_id, from_date, to_date)
        if not logs:
            return {}
        
        numeric_cols = [
            "calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g",
            "sodium_mg", "potassium_mg", "calcium_mg", "iron_mg", "magnesium_mg",
            "phosphorus_mg", "zinc_mg", "selenium_mcg", "cholesterol_mg",
            "saturated_fat_g", "monounsaturated_fat_g", "polyunsaturated_fat_g",
            "vitamin_a_mcg", "vitamin_c_mg", "vitamin_d_mcg", "vitamin_e_mg",
            "vitamin_k_mcg", "vitamin_b1_mg", "vitamin_b2_mg", "vitamin_b3_mg",
            "vitamin_b6_mg", "vitamin_b12_mcg",
        ]
        
        totals = {}
        for col in numeric_cols:
            totals[col] = sum(l.get(col, 0) or 0 for l in logs)
        
        days = len(set(l.get("log_date", "") for l in logs)) or 1
        
        return {
            "totals": totals,
            "averages": {k: round(v / days, 2) for k, v in totals.items()},
            "days_count": days,
            "entries_count": len(logs),
        }


# Global singleton
store = SupabaseStore()
