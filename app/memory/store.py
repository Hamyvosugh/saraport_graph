# In-memory store for MVP
from typing import Any


class InMemoryStore:
    def __init__(self):
        self._profiles: dict[str, dict[str, Any]] = {}
        self._plans: dict[str, dict[str, Any]] = {}
        self._logs: dict[str, list[dict[str, Any]]] = {}
        self._memories: dict[str, dict[str, Any]] = {}

    def get_profile(self, user_id: str) -> dict[str, Any]:
        return self._profiles.get(user_id, {})

    def set_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        self._profiles[user_id] = profile

    def get_plan(self, user_id: str) -> dict[str, Any]:
        return self._plans.get(user_id, {})

    def set_plan(self, user_id: str, plan: dict[str, Any]) -> None:
        self._plans[user_id] = plan

    def get_logs(self, user_id: str) -> list[dict[str, Any]]:
        return self._logs.get(user_id, [])

    def add_log(self, user_id: str, log: dict[str, Any]) -> None:
        if user_id not in self._logs:
            self._logs[user_id] = []
        self._logs[user_id].append(log)

    def get_memory(self, user_id: str) -> dict[str, Any]:
        return self._memories.get(user_id, {})

    def set_memory(self, user_id: str, memory: dict[str, Any]) -> None:
        self._memories[user_id] = memory


# Global singleton store
store = InMemoryStore()