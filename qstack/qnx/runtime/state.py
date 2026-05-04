from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class QNXState:
    """Deterministic runtime state container."""

    data: Dict[str, Any] = field(default_factory=dict)

    def update(self, key: str, value: Any) -> None:
        new_state = dict(self.data)
        new_state[key] = value
        self.data = new_state

    def read(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
