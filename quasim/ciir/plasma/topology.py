r"""Field topology analyzer: interface manifold 𝕀 of the 2D RMHD system.

[V] = established physics. [D] = derived. [H] = hypothesis.

The CIIR "interface" object 𝕀 is identified with the topologically distinguished
subset of the magnetic field where field-line topology can change under finite
resistivity. In 2D this consists of:

- **X-points / O-points**: critical points of the flux function ``ψ`` where
  ``∇ψ = 0``. Classification is by the sign of the Hessian determinant
  :math:`\det H_\psi = \psi_{xx}\psi_{yy} - \psi_{xy}^2` [V]:

  - :math:`\det H_\psi > 0` → O-point (elliptic, magnetic island center).
  - :math:`\det H_\psi < 0` → X-point (hyperbolic, reconnection site).

- **Current sheets**: connected regions where :math:`|J_z|` exceeds a
  threshold (typically a multiple of the RMS current). These are the loci
  where Sweet–Parker / plasmoid reconnection occurs [V].

- **Separatrices**: level sets of ``ψ`` passing through X-points. Not
  computed pointwise here (would require contour tracing); instead we expose
  the X-point list, which is sufficient for interface bookkeeping.

References: Biskamp, *Magnetic Reconnection in Plasmas* (CUP, 2000) [V];
Priest & Forbes, *Magnetic Reconnection* (CUP, 2000) [V].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .mhd import MHDState, ddx, ddy

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


# ---------------------------------------------------------------------------
# Critical points of ψ (X-points and O-points)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticalPoint:
    """A critical point of ``ψ``."""

    i: int
    j: int
    x: float
    y: float
    psi: float
    Jz: float
    kind: Literal["X", "O"]
    det_hessian: float


def _hessian_components(psi: FloatArray, dx: float, dy: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(ψ_xx, ψ_yy, ψ_xy)`` with periodic boundaries [V]."""
    psi_xx = (np.roll(psi, -1, axis=0) - 2.0 * psi + np.roll(psi, 1, axis=0)) / (dx * dx)
    psi_yy = (np.roll(psi, -1, axis=1) - 2.0 * psi + np.roll(psi, 1, axis=1)) / (dy * dy)
    psi_xy = ddy(ddx(psi, dx), dy)
    return psi_xx, psi_yy, psi_xy


def find_critical_points(state: MHDState, grad_tol: float = 1.0e-2) -> list[CriticalPoint]:
    r"""Locate critical points of ``ψ`` on the grid [V].

    A grid point is flagged as a candidate critical point iff
    :math:`(\partial_x \psi, \partial_y \psi)` change sign in *both* directions
    in the local 3-point stencil — a discrete analogue of the implicit-function
    criterion for an interior zero of :math:`\nabla \psi`.

    Classification uses the Hessian determinant [V]: ``det > 0`` → O-point,
    ``det < 0`` → X-point. ``det == 0`` (degenerate) points are skipped.

    Parameters
    ----------
    grad_tol : float
        Relative tolerance: only flag candidates whose minimal stencil gradient
        magnitude is below ``grad_tol * max(|∇ψ|)``. Filters out spurious hits
        from finite-precision sign flips far from a true zero.
    """
    g = state.grid
    psi = state.psi
    psi_x = ddx(psi, g.dx)
    psi_y = ddy(psi, g.dy)
    grad_norm = np.sqrt(psi_x * psi_x + psi_y * psi_y)
    threshold = grad_tol * float(grad_norm.max() + 1.0e-30)

    # Sign-change pattern using rolls: a true zero of ψ_x lies between i-1 and i+1
    # if their signs differ; same for ψ_y in the j direction.
    sx_left = np.sign(np.roll(psi_x, 1, axis=0))
    sx_right = np.sign(np.roll(psi_x, -1, axis=0))
    sy_down = np.sign(np.roll(psi_y, 1, axis=1))
    sy_up = np.sign(np.roll(psi_y, -1, axis=1))

    cand = (sx_left * sx_right < 0) & (sy_down * sy_up < 0) & (grad_norm < threshold)

    psi_xx, psi_yy, psi_xy = _hessian_components(psi, g.dx, g.dy)
    detH = psi_xx * psi_yy - psi_xy * psi_xy
    Jz = state.Jz()

    out: list[CriticalPoint] = []
    idxs = np.argwhere(cand)
    for i, j in idxs:
        d = float(detH[i, j])
        if d == 0.0:
            continue
        kind: Literal["X", "O"] = "O" if d > 0.0 else "X"
        out.append(
            CriticalPoint(
                i=int(i),
                j=int(j),
                x=float(i) * g.dx,
                y=float(j) * g.dy,
                psi=float(psi[i, j]),
                Jz=float(Jz[i, j]),
                kind=kind,
                det_hessian=d,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Current sheets
# ---------------------------------------------------------------------------


@dataclass
class CurrentSheet:
    """A connected current-sheet region.

    Attributes
    ----------
    mask : FloatArray
        Boolean mask (as float 0/1) of the sheet on the grid.
    n_points : int
    area : float
        Physical area in code units.
    Jz_max : float
    Jz_mean : float
    centroid : tuple[float, float]
        ``(x_c, y_c)`` in physical units.
    """

    mask: FloatArray
    n_points: int
    area: float
    Jz_max: float
    Jz_mean: float
    centroid: tuple[float, float]


def _label_connected(mask: NDArray[np.bool_]) -> tuple[IntArray, int]:
    """4-connected labeling on a *periodic* 2D mask using union–find.

    Returns ``(labels, n_components)``. ``labels`` is 0 for background and
    1..n_components for foreground components. Implemented locally to avoid
    requiring scipy as a hard dependency.
    """
    nx, ny = mask.shape
    labels = np.zeros((nx, ny), dtype=np.int64)
    parent: list[int] = [0]  # parent[0] = 0 sentinel; real labels start at 1

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    next_label = 1
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                continue
            up = labels[i, (j - 1) % ny]
            left = labels[(i - 1) % nx, j]
            if up == 0 and left == 0:
                labels[i, j] = next_label
                parent.append(next_label)
                next_label += 1
            elif up != 0 and left == 0:
                labels[i, j] = up
            elif up == 0 and left != 0:
                labels[i, j] = left
            else:
                m = min(up, left)
                labels[i, j] = m
                union(up, left)

    # Second pass: also stitch periodic seams (i=0 ↔ i=nx-1, j=0 ↔ j=ny-1).
    for j in range(ny):
        if mask[0, j] and mask[nx - 1, j]:
            union(int(labels[0, j]), int(labels[nx - 1, j]))
    for i in range(nx):
        if mask[i, 0] and mask[i, ny - 1]:
            union(int(labels[i, 0]), int(labels[i, ny - 1]))

    # Flatten and renumber consecutively.
    remap: dict[int, int] = {0: 0}
    n_comp = 0
    for lab in range(1, next_label):
        root = find(lab)
        if root not in remap:
            n_comp += 1
            remap[root] = n_comp
    out = np.zeros_like(labels)
    for i in range(nx):
        for j in range(ny):
            lab = int(labels[i, j])
            if lab != 0:
                out[i, j] = remap[find(lab)]
    return out, n_comp


def find_current_sheets(
    state: MHDState,
    threshold_factor: float = 3.0,
    min_size: int = 8,
) -> list[CurrentSheet]:
    r"""Identify connected current-sheet regions.

    A grid point is flagged when :math:`|J_z| > \mathrm{threshold\_factor}\times
    J_{rms}`. Connected components below ``min_size`` cells are discarded as
    noise.

    The threshold-on-RMS criterion is heuristic [D]; typical values
    ``threshold_factor ∈ [2, 5]`` track current-sheet structures in standard
    tearing/plasmoid simulations.
    """
    Jz = state.Jz()
    Jrms = float(np.sqrt(np.mean(Jz * Jz)) + 1.0e-30)
    mask = np.abs(Jz) > threshold_factor * Jrms
    labels, n_comp = _label_connected(mask)
    g = state.grid
    cell_area = g.dx * g.dy

    sheets: list[CurrentSheet] = []
    for k in range(1, n_comp + 1):
        comp = labels == k
        n_pts = int(comp.sum())
        if n_pts < min_size:
            continue
        Jz_local = Jz[comp]
        idxs = np.argwhere(comp)
        # Periodic-aware centroid via circular mean on each axis.
        xs = idxs[:, 0] * g.dx
        ys = idxs[:, 1] * g.dy
        cx = float(_circular_mean(xs, g.Lx))
        cy = float(_circular_mean(ys, g.Ly))
        sheets.append(
            CurrentSheet(
                mask=comp.astype(np.float64),
                n_points=n_pts,
                area=n_pts * cell_area,
                Jz_max=float(np.max(np.abs(Jz_local))),
                Jz_mean=float(np.mean(np.abs(Jz_local))),
                centroid=(cx, cy),
            )
        )
    sheets.sort(key=lambda s: s.area, reverse=True)
    return sheets


def _circular_mean(values: FloatArray, period: float) -> float:
    """Mean on a circle of circumference ``period`` (avoids wraparound bias) [V]."""
    if values.size == 0:
        return 0.0
    theta = 2.0 * np.pi * values / period
    s = float(np.sin(theta).mean())
    c = float(np.cos(theta).mean())
    th = np.arctan2(s, c)
    if th < 0.0:
        th += 2.0 * np.pi
    return th * period / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# Reconnection rate
# ---------------------------------------------------------------------------


def reconnection_rate_field(state: MHDState) -> FloatArray:
    r"""Local reconnection-rate proxy :math:`\mathcal R(x, y) = \eta J_z` [V].

    In 2D resistive MHD, the reconnection electric field at an X-point is
    :math:`E_z = \eta J_z` (Ohm's law projected on :math:`\hat z`). The full
    field is returned so callers may probe it at X-points.
    """
    return state.eta * state.Jz()


@dataclass
class TopologySnapshot:
    """Aggregate interface snapshot 𝕀(t)."""

    t: float
    x_points: list[CriticalPoint]
    o_points: list[CriticalPoint]
    sheets: list[CurrentSheet]
    reconnection_rate_at_X: float
    """Mean :math:`|\\eta J_z|` averaged over X-points (0 if none)."""

    n_X: int
    n_O: int


def topology_snapshot(
    state: MHDState,
    grad_tol: float = 1.0e-2,
    sheet_threshold: float = 3.0,
    sheet_min_size: int = 8,
) -> TopologySnapshot:
    """Compute a full topology snapshot of the current state."""
    crits = find_critical_points(state, grad_tol=grad_tol)
    xs = [c for c in crits if c.kind == "X"]
    op = [c for c in crits if c.kind == "O"]
    sheets = find_current_sheets(state, threshold_factor=sheet_threshold, min_size=sheet_min_size)
    if xs:
        rate = float(np.mean([abs(state.eta * c.Jz) for c in xs]))
    else:
        rate = 0.0
    return TopologySnapshot(
        t=state.t,
        x_points=xs,
        o_points=op,
        sheets=sheets,
        reconnection_rate_at_X=rate,
        n_X=len(xs),
        n_O=len(op),
    )


__all__ = [
    "CriticalPoint",
    "CurrentSheet",
    "TopologySnapshot",
    "find_critical_points",
    "find_current_sheets",
    "reconnection_rate_field",
    "topology_snapshot",
]
