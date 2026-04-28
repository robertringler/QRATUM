"""qnx bridge: distributed CIIR evolution executor.

Maps CIIR state evolution to the qnx distributed execution engine,
enabling parallel evolution across multiple compute nodes.

QRATUM subsystem: qnx (distributed execution)
CIIR mapping: evolution loop ↔ distributed step execution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quasim.ciir.config import CIIRConfig
from quasim.ciir.evolution import IntegrationMethod, evolve
from quasim.ciir.loss import CIIRLoss
from quasim.ciir.theory import CIIRTheory


@dataclass
class QNXEvolutionTask:
    """A single CIIR evolution task for qnx distributed execution.

    Parameters
    ----------
    task_id : str
        Unique identifier.
    config : CIIRConfig
        Evolution configuration.
    shard_index : int
        Which batch shard this task handles.
    n_shards : int
        Total number of parallel shards.
    """

    task_id: str
    config: CIIRConfig
    shard_index: int = 0
    n_shards: int = 1

    def shard_batch_size(self) -> int:
        """Batch size for this shard."""
        total = self.config.batch_size
        base = total // self.n_shards
        extra = 1 if self.shard_index < (total % self.n_shards) else 0
        return base + extra

    def to_qnx_payload(self) -> dict[str, Any]:
        """Serialise as qnx task payload."""
        return {
            "task_id": self.task_id,
            "task_type": "ciir_evolution",
            "shard_index": self.shard_index,
            "n_shards": self.n_shards,
            "config": self.config.to_dict(),
        }


@dataclass
class QNXDistributedEvolver:
    """Orchestrate distributed CIIR evolution via qnx.

    Splits the batch across shards, runs evolution on each,
    and aggregates results.

    Parameters
    ----------
    theory : CIIRTheory
        The CIIR theory tuple (shared across shards).
    config : CIIRConfig
        Evolution configuration.
    n_shards : int
        Number of parallel execution shards.
    """

    theory: CIIRTheory
    config: CIIRConfig
    n_shards: int = 1

    def create_tasks(self) -> list[QNXEvolutionTask]:
        """Create one task per shard."""
        tasks = []
        for s in range(self.n_shards):
            task = QNXEvolutionTask(
                task_id=f"ciir-shard-{s}",
                config=self.config,
                shard_index=s,
                n_shards=self.n_shards,
            )
            tasks.append(task)
        return tasks

    def run_local(self) -> dict[str, Any]:
        """Execute all shards locally (for testing).

        In production, each shard would be dispatched to a qnx
        executor node.
        """
        rng = np.random.default_rng(self.config.seed)
        rho_full = self.theory.manifold.density_matrix(
            self.theory.manifold.random_state(batch=self.config.batch_size, rng=rng)
        )

        loss_fn = CIIRLoss(
            theory=self.theory,
            entropy_weight=self.config.entropy_weight,
        )

        method_map = {
            "gradient": IntegrationMethod.GRADIENT,
            "projected_gradient": IntegrationMethod.PROJECTED_GRADIENT,
            "strang_splitting": IntegrationMethod.STRANG_SPLITTING,
        }
        method = method_map.get(self.config.method, IntegrationMethod.PROJECTED_GRADIENT)

        # Split batch across shards
        shard_results = []
        offset = 0
        for s in range(self.n_shards):
            task = QNXEvolutionTask(
                task_id=f"ciir-shard-{s}",
                config=self.config,
                shard_index=s,
                n_shards=self.n_shards,
            )
            bs = task.shard_batch_size()
            rho_shard = rho_full[offset : offset + bs]
            offset += bs

            result = evolve(
                loss_fn,
                rho_shard,
                n_steps=self.config.n_steps,
                lr=self.config.lr,
                method=method,
                tol=self.config.tol,
                record_every=self.config.record_every,
            )
            shard_results.append(
                {
                    "shard": s,
                    "converged": result.converged,
                    "n_steps": result.n_steps,
                    "final_loss": result.losses[-1] if result.losses else None,
                }
            )

        return {
            "n_shards": self.n_shards,
            "shards": shard_results,
            "all_converged": all(r["converged"] for r in shard_results),
        }
