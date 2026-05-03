"""config.py — single config-profile loader.

The plan calls for **one config** (``pipeline_config.yaml``) replacing the
seven ``pipeline_config_*.yaml`` variants, exposing named profiles
(``quick``, ``medium``, ``strong``, ``full_fast``, ``full_report``).

This module ships those profiles as in-process defaults so the framework
boots without a config file present, and provides ``load_profile`` for
overriding any field from a YAML file when one *is* available.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class PipelineProfile:
    """Profile-level knobs that affect every run.

    Profiles are deliberately small — they describe operational posture
    (how long to run, how aggressively to seed perturbations, whether to
    enforce strict invariants), not domain physics.  Domain knobs live in
    domain packages and are reachable through the backend, not here.
    """

    name: str
    n_steps: int
    seed: int
    strict_invariants: bool = True
    falsification_seed_count: int = 1
    record_metrics: bool = True

    def with_overrides(self, **kwargs: Any) -> PipelineProfile:
        """Return a copy with selected fields overridden (immutable update)."""
        return replace(self, **kwargs)


#: Canonical built-in profiles.  These are the named profiles cited in the
#: plan; the values are conservative defaults suitable for CI smoke runs.
PROFILES: Dict[str, PipelineProfile] = {
    "quick": PipelineProfile(name="quick", n_steps=8, seed=0, falsification_seed_count=1),
    "medium": PipelineProfile(name="medium", n_steps=32, seed=0, falsification_seed_count=2),
    "strong": PipelineProfile(name="strong", n_steps=128, seed=0, falsification_seed_count=4),
    "full_fast": PipelineProfile(name="full_fast", n_steps=256, seed=0, falsification_seed_count=4),
    "full_report": PipelineProfile(
        name="full_report", n_steps=1024, seed=0, falsification_seed_count=8
    ),
}

#: The profile used when none is specified.  ``quick`` is the only profile
#: that is safe to run in every CI lane unconditionally.
DEFAULT_PROFILE: str = "quick"


def load_profile(
    name: str = DEFAULT_PROFILE,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    yaml_path: Optional[Path] = None,
) -> PipelineProfile:
    """Resolve a named profile, optionally overlaying YAML or kw overrides.

    Resolution order (lowest → highest priority):

    1. The built-in :data:`PROFILES` table.
    2. The ``profiles.<name>`` block in ``yaml_path`` if provided and
       parseable (PyYAML is an optional dependency; if absent, the YAML
       layer is silently skipped to keep imports cheap).
    3. The ``overrides`` mapping, if given.

    Unknown profile names raise :class:`KeyError`.
    """
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; available: {sorted(PROFILES)}")
    profile = PROFILES[name]

    if yaml_path is not None and Path(yaml_path).exists():
        try:
            import yaml  # type: ignore[import-not-found]

            data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
            block = ((data.get("profiles") or {}).get(name)) or {}
            if isinstance(block, Mapping) and block:
                profile = profile.with_overrides(
                    **{k: v for k, v in block.items() if k in profile.__dataclass_fields__}
                )
        except Exception:
            # YAML is best-effort; never fail config resolution because the
            # optional dep is absent or the file is malformed.
            pass

    if overrides:
        profile = profile.with_overrides(
            **{k: v for k, v in overrides.items() if k in profile.__dataclass_fields__}
        )
    return profile


__all__ = [
    "PipelineProfile",
    "PROFILES",
    "DEFAULT_PROFILE",
    "load_profile",
]
