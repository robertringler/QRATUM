"""Loading and hashing of FLUXA machine-readable configuration.

Configuration is the single source of truth for every number the simulation
uses. Nothing is hard-coded in the engine. Two hashes are derived here and
flow into the QRADLE provenance chain:

* ``config_hash``  -- SHA-256 over the raw configuration document as loaded.
* ``model_hash``   -- SHA-256 over the validated :class:`EnergySystem`.

They differ deliberately: ``config_hash`` changes if a comment or key order
changes, ``model_hash`` only if the physics changes.

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fluxa.model import (
    Battery,
    Bus,
    EconomicAssumptions,
    EnergySystem,
    FrequencyModel,
    Generator,
    GeneratorKind,
    Interconnection,
    Line,
    ReservePolicy,
)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_SYSTEM_CONFIG = CONFIG_DIR / "system_6bus.json"
DEFAULT_SCENARIO_CONFIG = CONFIG_DIR / "scenarios.json"
DEFAULT_RUN_CONFIG = CONFIG_DIR / "run_config.json"

SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0.0"})


class ConfigError(ValueError):
    """Raised when a configuration document is malformed or unsupported."""


@dataclass(frozen=True)
class LoadedSystem:
    """An :class:`EnergySystem` plus the provenance of the document it came from."""

    system: EnergySystem
    config_path: str
    config_hash: str
    raw: dict[str, Any]

    @property
    def model_hash(self) -> str:
        return self.system.model_hash()


def canonical_hash(payload: Any) -> str:
    """SHA-256 over a canonical JSON encoding of ``payload``.

    Keys are sorted and separators are tight, so the digest depends on
    content only -- never on dict insertion order or whitespace.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, GeneratorKind):
        return obj.value
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def _require(doc: dict[str, Any], key: str, path: str) -> Any:
    if key not in doc:
        raise ConfigError(f"{path}: missing required key '{key}'")
    return doc[key]


def load_system(path: str | Path = DEFAULT_SYSTEM_CONFIG) -> LoadedSystem:
    """Load, validate and hash an energy-system configuration document.

    Raises:
        ConfigError: on a missing file, bad JSON, unsupported schema version,
            or any physical inconsistency detected by the model validators.
    """
    cfg_path = Path(path)
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read system config {cfg_path}: {exc}") from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{cfg_path}: invalid JSON: {exc}") from exc

    version = doc.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigError(
            f"{cfg_path}: unsupported schema_version {version!r}; "
            f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    name = str(cfg_path)
    try:
        system = EnergySystem(
            system_id=_require(doc, "system_id", name),
            description=doc.get("description", ""),
            buses=tuple(Bus(**b) for b in _require(doc, "buses", name)),
            lines=tuple(Line(**ln) for ln in _require(doc, "lines", name)),
            generators=tuple(
                Generator(**{**g, "kind": GeneratorKind(g["kind"])})
                for g in _require(doc, "generators", name)
            ),
            batteries=tuple(Battery(**b) for b in doc.get("batteries", [])),
            interconnections=tuple(
                Interconnection(**i) for i in doc.get("interconnections", [])
            ),
            peak_load_mw=float(_require(doc, "peak_load_mw", name)),
            economics=EconomicAssumptions(**doc.get("economics", {})),
            reserve=ReservePolicy(**doc.get("reserve", {})),
            frequency=FrequencyModel(**doc.get("frequency", {})),
        )
    except (TypeError, KeyError) as exc:
        raise ConfigError(f"{cfg_path}: malformed entry: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"{cfg_path}: physically inconsistent system: {exc}") from exc

    return LoadedSystem(
        system=system,
        config_path=str(cfg_path),
        config_hash=canonical_hash(doc),
        raw=doc,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Load an arbitrary FLUXA JSON document with a clear error on failure."""
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{p}: invalid JSON: {exc}") from exc
