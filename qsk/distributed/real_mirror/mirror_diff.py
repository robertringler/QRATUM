"""Compare mirrored state against current cluster world model."""

from __future__ import annotations

from typing import Dict

from qsk.distributed.real_mirror.mirror_state import MirrorState


def diff(mirror: MirrorState, world: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    deltas: Dict[str, Dict[str, object]] = {}
    for domain, mirror_state in mirror.domains.items():
        cluster_state = world.get(domain, {})
        domain_diff: Dict[str, object] = {}
        for key in sorted(set(mirror_state.keys()) | set(cluster_state.keys())):
            if mirror_state.get(key) != cluster_state.get(key):
                domain_diff[key] = {
                    "mirror": mirror_state.get(key),
                    "cluster": cluster_state.get(key),
                }
        if domain_diff:
            deltas[domain] = domain_diff
    return deltas
