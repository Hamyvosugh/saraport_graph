from typing import TypedDict


class HealthState(TypedDict):
    user_id: str
    input: dict
    messages: list
    profile: dict
    plan: dict
    logs: list
    memory: dict
    output: dict