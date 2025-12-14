import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Signer:
    """
    Deterministic SHA-256 based MAC. Suitable for integrity checks
    across the Q-Stack modules. Keys should be derived via KeyManager.
    """

    key: str

    def sign(self, message: Any) -> str:
        payload = f"{self.key}:{repr(message)}".encode()
        return hashlib.sha256(payload).hexdigest()

    def verify(self, message: Any, signature: str) -> bool:
        return signature == self.sign(message)
