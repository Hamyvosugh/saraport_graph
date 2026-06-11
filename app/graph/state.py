from typing import TypedDict, Optional


class HealthState(TypedDict, total=False):
    user_id: str
    input: dict
    messages: list
    profile: dict
    plan: dict
    logs: list
    log_entries: list  # new flexible log_entries from the new table
    memory: dict
    output: dict
    detected_logs: dict
    pending_questions: Optional[list]  # questions for the user to resolve ambiguities
