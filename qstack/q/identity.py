from dataclasses import dataclass


@dataclass(frozen=True)
class QIdentity:
    """Immutable identity record."""

    name: str
    key: str

    def to_dict(self) -> dict:
        return {"name": self.name, "key": self.key}
