"""config.py — SDE-specific config sub-block.

Exposed as :class:`SDEConfig` and the ``"sde"`` sub-key the spec
requires under each ``PROFILES`` entry.  We do *not* mutate the
existing :class:`qratum_framework.config.PipelineProfile` dataclass:
that would break every consumer that imports it.  Instead, we keep
the SDE block as a side-table (:data:`SDE_PROFILES`) keyed by the
same profile name and provide :func:`load_sde_config` to resolve it
with the same precedence rules as :func:`qratum_framework.config.load_profile`
(built-in defaults → optional YAML overlay → kw overrides).

Built-in profile values mirror the spec defaults exactly; per-profile
deviation is reserved for tuning and currently uniform across all
profiles for safety.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from qratum_framework.config import DEFAULT_PROFILE, PROFILES


@dataclass(frozen=True)
class SDEConfig:
    """Configuration sub-block for the Streaming Drift Engine.

    Field semantics mirror §FORMAL SPECIFICATION:

    * ``window_depth`` — rolling buffer depth ``L``.
    * ``min_window`` — minimum population before scoring begins.
    * ``scorer`` — ``"kl"`` or ``"mmd"``; selects the concrete scorer
      built by :func:`build_scorer`.
    * ``theta_low`` / ``theta_high`` / ``theta_halt`` — alarm cuts.
    * ``n_debounce`` — consecutive readings required to escalate.
    """

    window_depth: int = 256
    min_window: int = 32
    scorer: str = "kl"
    theta_low: float = 0.05
    theta_high: float = 0.20
    theta_halt: float = 0.50
    n_debounce: int = 3

    def __post_init__(self) -> None:
        if self.window_depth <= 0:
            raise ValueError(f"window_depth must be positive, got {self.window_depth}")
        if self.min_window <= 0:
            raise ValueError(f"min_window must be positive, got {self.min_window}")
        if self.min_window > self.window_depth:
            raise ValueError(
                f"min_window ({self.min_window}) must not exceed "
                f"window_depth ({self.window_depth})"
            )
        if self.scorer not in ("kl", "mmd"):
            raise ValueError(f"scorer must be 'kl' or 'mmd', got {self.scorer!r}")
        if not (0.0 <= self.theta_low <= self.theta_high <= self.theta_halt):
            raise ValueError(
                "thresholds must satisfy 0 ≤ theta_low ≤ theta_high ≤ theta_halt; "
                f"got low={self.theta_low}, high={self.theta_high}, "
                f"halt={self.theta_halt}"
            )
        if self.n_debounce < 1:
            raise ValueError(f"n_debounce must be >= 1, got {self.n_debounce}")

    def with_overrides(self, **kwargs: Any) -> "SDEConfig":
        """Return a copy with selected fields overridden (immutable update)."""
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_depth": int(self.window_depth),
            "min_window": int(self.min_window),
            "scorer": self.scorer,
            "theta_low": float(self.theta_low),
            "theta_high": float(self.theta_high),
            "theta_halt": float(self.theta_halt),
            "n_debounce": int(self.n_debounce),
        }


#: Built-in SDE sub-block per pipeline profile.  Keys mirror
#: :data:`qratum_framework.config.PROFILES` exactly so that
#: ``load_sde_config(name)`` always resolves.
SDE_PROFILES: Dict[str, SDEConfig] = {
    name: SDEConfig() for name in PROFILES
}


def load_sde_config(
    name: str = DEFAULT_PROFILE,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    yaml_path: Optional[Path] = None,
) -> SDEConfig:
    """Resolve the SDE sub-block for a named profile.

    Resolution order (lowest → highest priority):

    1. The built-in :data:`SDE_PROFILES` table.
    2. The ``profiles.<name>.sde`` block in ``yaml_path`` if provided
       and parseable (PyYAML is optional; absent ⇒ skip).
    3. The ``overrides`` mapping, if given.

    Unknown profile names raise :class:`KeyError`.
    """
    if name not in SDE_PROFILES:
        raise KeyError(
            f"unknown profile {name!r}; available: {sorted(SDE_PROFILES)}"
        )
    cfg = SDE_PROFILES[name]

    if yaml_path is not None and Path(yaml_path).exists():
        try:
            import yaml  # type: ignore[import-not-found]

            data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
            block = (
                ((data.get("profiles") or {}).get(name) or {}).get("sde") or {}
            )
            if isinstance(block, Mapping) and block:
                cfg = cfg.with_overrides(
                    **{
                        k: v
                        for k, v in block.items()
                        if k in cfg.__dataclass_fields__
                    }
                )
        except Exception:
            # YAML is best-effort; never fail config resolution because
            # the optional dep is absent or the file is malformed.
            pass

    if overrides:
        cfg = cfg.with_overrides(
            **{
                k: v
                for k, v in overrides.items()
                if k in cfg.__dataclass_fields__
            }
        )
    return cfg


# ---------------------------------------------------------------------------
# Builders — instantiate SDE primitives from an :class:`SDEConfig`.
# ---------------------------------------------------------------------------


def build_scorer(cfg: SDEConfig):
    """Construct the concrete scorer named by ``cfg.scorer``."""
    # Imported here to avoid a circular import at module load time.
    from qratum_framework.sde.scorers import KLDriftScorer, MMDDriftScorer

    if cfg.scorer == "kl":
        return KLDriftScorer()
    if cfg.scorer == "mmd":
        return MMDDriftScorer()
    raise ValueError(f"unknown scorer {cfg.scorer!r}")


def build_buffer(cfg: SDEConfig):
    """Construct a :class:`RollingWindowBuffer` from ``cfg``."""
    from qratum_framework.sde.buffer import RollingWindowBuffer

    return RollingWindowBuffer(depth=cfg.window_depth, min_window=cfg.min_window)


def build_evaluator(cfg: SDEConfig):
    """Construct an :class:`AlarmEvaluator` from ``cfg``."""
    from qratum_framework.sde.evaluator import AlarmEvaluator, AlarmThresholds

    th = AlarmThresholds(
        theta_low=cfg.theta_low,
        theta_high=cfg.theta_high,
        theta_halt=cfg.theta_halt,
    )
    return AlarmEvaluator(th, n_debounce=cfg.n_debounce)


__all__ = [
    "SDEConfig",
    "SDE_PROFILES",
    "load_sde_config",
    "build_scorer",
    "build_buffer",
    "build_evaluator",
]
