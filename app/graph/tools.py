"""Tools for the Coach Agent — read/write from Supabase."""

import json
from typing import Optional
from langchain_core.tools import tool
from app.memory.supabase_store import store as db


# ---------------------------------------------------------------------------
@tool
def get_user_profile(user_id: str) -> str:
    """دریافت پروفایل کاربر از دیتابیس
    Get the user's health profile including age, gender, height, weight, goals, etc."""
    profile = db.get_profile(user_id)
    if not profile:
        return "کاربر هنوز پروفایل خود را تکمیل نکرده است. لطفاً از کاربر بخواهید اطلاعات خود را وارد کند."
    return json.dumps(profile, ensure_ascii=False, default=str)


@tool
def get_user_plan(user_id: str) -> str:
    """دریافت برنامه روزانه کاربر (اهداف کالری، پروتئین، قدم‌ها و غیره)
    Get the user's daily plan with calorie, protein, steps, water, sleep targets."""
    plan = db.get_plan(user_id)
    if not plan:
        return "هنوز برنامه‌ای برای کاربر تنظیم نشده است. باید از کاربر اطلاعات بیشتری بپرسی و برنامه بریزی."
    return json.dumps(plan, ensure_ascii=False, default=str)


@tool
def get_user_logs(user_id: str, days: int = 7, category: str = None) -> str:
    """دریافت لاگ‌های روزانه اخیر کاربر از جدول log_entries
    Get the user's recent log entries (weight, steps, water, sleep, etc.) for the last N days.
    Args:
        user_id: شناسه کاربر
        days: تعداد روزهای اخیر (پیش‌فرض ۷)
        category: فیلتر بر اساس دسته - 'measurement' برای اندازه‌گیری‌ها یا 'activity' برای فعالیت‌ها
    """
    logs = db.get_log_entries(user_id, days=days, category=category)
    if not logs:
        return "هنوز هیچ لاگ روزانه‌ای ثبت نشده است. کاربر را تشویق کن که لاگ روزانه خود را ثبت کند."
    # Group by category for readability
    measurements = [l for l in logs if l.get("category") == "measurement"]
    activities = [l for l in logs if l.get("category") == "activity"]
    result = {
        "total_entries": len(logs),
        "days_covered": len(set(l.get("log_date") for l in logs)),
    }
    if measurements:
        result["measurements"] = measurements
    if activities:
        result["activities"] = activities
        # Activity totals
        totals = {}
        for a in activities:
            t = a.get("entry_type", "")
            totals[t] = totals.get(t, 0) + (a.get("value", 0) or 0)
        result["activity_totals"] = totals
    return json.dumps(result, ensure_ascii=False, default=str)


@tool
def get_user_memory(user_id: str) -> str:
    """دریافت حافظه بلندمدت مربی از مکالمات قبلی
    Get the coach's long-term conversation memory/summary."""
    mem = db.get_memory(user_id)
    if not mem:
        return "هنوز حافظه‌ای ذخیره نشده است. این اولین گفتگوی مربی با کاربر است."
    return json.dumps(mem, ensure_ascii=False, default=str)


@tool
def get_chat_history(user_id: str, last_n: int = 10) -> str:
    """دریافت تاریخچه اخیر گفتگوی کاربر با مربی
    Get the last N messages between the user and coach."""
    msgs = db.get_messages(user_id, limit=last_n * 2)
    if not msgs:
        return "تاریخچه پیامی موجود نیست."
    simplified = [{"role": m.get("role"), "content": m.get("content")} for m in msgs[-last_n * 2:]]
    return json.dumps(simplified, ensure_ascii=False, default=str)


@tool
def save_coach_memory(user_id: str, summary: str) -> str:
    """ذخیره خلاصه‌ای از وضعیت کاربر در حافظه بلندمدت مربی
    Save a summary of the user's progress and status to long-term memory.
    Args:
        user_id: شناسه کاربر
        summary: خلاصه وضعیت کاربر (به فارسی یا انگلیسی)"""
    db.set_memory(user_id, {"summary": summary, "updated_at": "now"})
    return "✅ حافظه مربی با موفقیت ذخیره شد."


@tool
def save_message(user_id: str, role: str, content: str) -> str:
    """ذخیره یک پیام در تاریخچه گفتگو
    Save a message to the chat history.
    Args:
        user_id: شناسه کاربر
        role: 'user' یا 'assistant'
        content: متن پیام"""
    db.add_message(user_id, role, content)
    return "✅ پیام ذخیره شد."


@tool
def analyze_food_note(user_id: str, note: str) -> str:
    """تحلیل غذایی که کاربر خورده است بر اساس توضیحات متنی
    Analyze a food description and estimate calories and macros.
    Args:
        user_id: شناسه کاربر
        note: توضیحات غذای مصرفی (مثلاً 'مرغ و برنج و سالاد')"""
    from app.graph.nodes import food_vision_node
    from app.graph.state import HealthState

    state: HealthState = {
        "user_id": user_id,
        "input": {"note": note, "user_id": user_id},
        "messages": [],
        "profile": {},
        "plan": {},
        "logs": [],
        "memory": {},
        "output": {},
    }
    result_state = food_vision_node(state)
    output = result_state.get("output", {})
    db.add_food_log(user_id, {
        "foods_detected": output.get("foods_detected", []),
        "estimated_calories": output.get("estimated_calories", 0),
        "protein_g": output.get("protein_g", 0),
        "carbs_g": output.get("carbs_g", 0),
        "fat_g": output.get("fat_g", 0),
        "note": note,
    })
    return json.dumps(output, ensure_ascii=False, default=str)


@tool
def log_daily_progress(
    user_id: str,
    weight_kg: Optional[float] = None,
    waist_cm: Optional[float] = None,
    steps: Optional[int] = None,
    water_l: Optional[float] = None,
    sleep_h: Optional[float] = None,
    notes: str = "",
) -> str:
    """ثبت لاگ روزانه کاربر (قدم‌ها، آب، خواب، وزن و غیره)
    Log the user's daily progress metrics.
    Args:
        user_id: شناسه کاربر
        weight_kg: وزن به کیلوگرم
        waist_cm: دور کمر به سانتی‌متر
        steps: تعداد قدم‌ها
        water_l: میزان آب مصرفی به لیتر
        sleep_h: میزان خواب به ساعت
        notes: یادداشت‌های اضافی"""
    log_entry = {
        "weight_kg": weight_kg,
        "waist_cm": waist_cm,
        "steps": steps,
        "water_l": water_l,
        "sleep_h": sleep_h,
        "notes": notes,
    }
    log_entry = {k: v for k, v in log_entry.items() if v is not None}
    db.add_log(user_id, log_entry)
    return "✅ لاگ روزانه با موفقیت ثبت شد."


@tool
def request_plan_update(user_id: str, reason: str) -> str:
    """درخواست بازسازی برنامه روزانه کاربر (صدا زدن planner agent)
    Request regeneration of the user's daily plan based on new information or changed goals.
    Args:
        user_id: شناسه کاربر
        reason: دلیل بازسازی برنامه (مثلاً 'تغییر هدف', 'عدم پیشرفت', 'تغییر وزن')"""
    from app.graph.nodes import planner_node
    from app.graph.state import HealthState

    profile = db.get_profile(user_id)
    if not profile:
        return "❌ نمی‌توان برنامه را بازسازی کرد — پروفایل کاربر کامل نیست. از کاربر بخواه اطلاعات خود را تکمیل کند."

    state: HealthState = {
        "user_id": user_id,
        "input": profile,
        "messages": [],
        "profile": profile,
        "plan": {},
        "logs": db.get_logs(user_id, limit=7),
        "memory": db.get_memory(user_id),
        "output": {},
    }
    result_state = planner_node(state)
    new_plan = result_state.get("plan", {})
    db.set_plan(user_id, new_plan)
    db.set_memory(user_id, {"summary": f"Plan updated — reason: {reason}", "updated_at": "now"})
    return f"✅ برنامه کاربر با موفقیت بازسازی شد. دلیل: {reason}\n\nبرنامه جدید: {json.dumps(new_plan, ensure_ascii=False, default=str)}"


# ---------------------------------------------------------------------------
# All tools collection
# ---------------------------------------------------------------------------
ALL_TOOLS = [
    get_user_profile,
    get_user_plan,
    get_user_logs,
    get_user_memory,
    get_chat_history,
    save_coach_memory,
    save_message,
    analyze_food_note,
    log_daily_progress,
    request_plan_update,
]
