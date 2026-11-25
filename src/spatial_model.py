"""Velocity-aware control and an independently published expected-threat grid.

All positions use metres in a centred, fixed stadium coordinate system: x is
lengthwise and y is across the pitch. The xT grid is oriented towards +x;
set ``attack_direction=-1`` when the attacking team is playing towards -x.

Control is a deliberately lightweight *relative* time-to-intercept model,
not the full Spearman PPCF. It ignores ball travel, occlusion and pass options,
and its logistic scores have not been calibrated as probabilities of real
pass completion. In particular, incomplete broadcast-tracking frames can make
its output misleading; both teams must have at least one observed player.

xT grid: Karun Singh, open xT 12x8 v1,
https://karun.in/blog/data/open_xt_12x8_v1.json; methodology:
https://karun.in/blog/expected-threat.html. xT describes possession value at a
location, not a probability that an off-ball run scores a goal.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

FloatArray = NDArray[np.float64]

# The author's published grid is fetched on demand, rather than redistributed.
XT_GRID_URL = "https://karun.in/blog/data/open_xt_12x8_v1.json"
XT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "open_xt_12x8_v1.json"
_XT_GRID: FloatArray | None = None


def load_xt_grid(path: str | Path = XT_CACHE_PATH) -> FloatArray:
    """Load the published 12×8 grid, caching its original JSON on first use.

    Users can place the original JSON at ``path`` to use the model offline.
    A missing file triggers a download from the author's site. The upstream
    grid has its own terms and is not covered by this project's license.
    """
    cache = Path(path)
    if cache.exists():
        raw = cache.read_bytes()
    else:
        try:
            request = Request(XT_GRID_URL, headers={"User-Agent": "off-ball-space-valuation/1.0"})
            with urlopen(request, timeout=30) as response:
                raw = response.read()
        except (OSError, URLError) as exc:
            raise RuntimeError(f"Unable to fetch xT grid from {XT_GRID_URL}; place the JSON at {cache} to run offline") from exc
    try:
        grid = np.asarray(json.loads(raw), dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("xT grid must be a numeric 8×12 JSON array") from exc
    if grid.shape != (8, 12) or not np.isfinite(grid).all() or np.any((grid < 0) | (grid > 1)):
        raise ValueError("xT grid must be an 8×12 array of finite values in [0, 1]")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    return grid


def _xt_grid() -> FloatArray:
    global _XT_GRID
    if _XT_GRID is None:
        _XT_GRID = load_xt_grid()
    return _XT_GRID


@dataclass(frozen=True)
class ControlSurface:
    """A regularly spaced pitch grid; probability has shape (len(ys), len(xs))."""

    xs: FloatArray
    ys: FloatArray
    probability: FloatArray
    pitch_length: float
    pitch_width: float


@dataclass(frozen=True)
class ControlMetrics:
    """Integrated relative control: area in m², xT-weighted area in xT·m²."""

    controlled_area_m2: float
    threat_weighted_area_m2: float


def _players(positions: ArrayLike, velocities: ArrayLike | None) -> tuple[FloatArray, FloatArray]:
    pos = np.asarray(positions, dtype=float)
    if pos.size == 0:
        raise ValueError("At least one observed player per team is required")
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("Positions must have shape (n_players, 2)")
    if velocities is None:
        vel = np.zeros_like(pos)
    else:
        vel = np.asarray(velocities, dtype=float)
        if vel.shape != pos.shape:
            raise ValueError("Velocities must have the same shape as positions")
        # A missing velocity means stationary for the reaction extrapolation,
        # not a missing player. Never silently replace missing positions.
        vel = np.where(np.isfinite(vel), vel, 0.0)
    visible = np.isfinite(pos).all(axis=1)
    if not visible.any():
        raise ValueError("At least one observed player per team is required")
    return pos[visible], vel[visible]


def pitch_control(
    attack_positions: ArrayLike,
    defense_positions: ArrayLike,
    attack_velocities: ArrayLike | None = None,
    defense_velocities: ArrayLike | None = None,
    *,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    nx: int = 50,
    ny: int = 32,
    reaction_time: float = 0.7,
    max_speed: float = 5.0,
    time_scale: float = 0.45,
) -> ControlSurface:
    """Return an attacker-relative control score on a regular pitch grid.

    For each target, each player coasts for ``reaction_time`` seconds and then
    covers remaining distance at ``max_speed`` m/s. The quickest time per team
    is compared using sigmoid((defender_time - attacker_time) / time_scale).
    Current velocity is clipped to max_speed for reaction extrapolation.
    Only the nearest-time player on each side contributes. Scores sum to one
    across the two teams, but are not empirically calibrated probabilities.

    NaN player positions are omitted; missing velocity components are set to
    zero. Missing whole teams raise ValueError rather than fabricate control.
    """
    if not all(np.isfinite(v) and v > 0 for v in (pitch_length, pitch_width, max_speed, time_scale)):
        raise ValueError("Pitch dimensions, max_speed and time_scale must be finite and positive")
    if not np.isfinite(reaction_time) or reaction_time < 0:
        raise ValueError("reaction_time must be finite and non-negative")
    if not isinstance(nx, int) or not isinstance(ny, int) or nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive integers")
    a_pos, a_vel = _players(attack_positions, attack_velocities)
    d_pos, d_vel = _players(defense_positions, defense_velocities)
    xs = np.linspace(-pitch_length / 2 + pitch_length / (2 * nx), pitch_length / 2 - pitch_length / (2 * nx), nx)
    ys = np.linspace(-pitch_width / 2 + pitch_width / (2 * ny), pitch_width / 2 - pitch_width / (2 * ny), ny)
    xx, yy = np.meshgrid(xs, ys)
    targets = np.stack((xx, yy), axis=-1)

    def min_arrival(pos: FloatArray, vel: FloatArray) -> FloatArray:
        speeds = np.linalg.norm(vel, axis=1, keepdims=True)
        clipped_vel = vel * np.minimum(1.0, max_speed / np.maximum(speeds, 1e-12))
        reaction_position = pos + reaction_time * clipped_vel
        distances = np.linalg.norm(targets[None, :, :, :] - reaction_position[:, None, None, :], axis=-1)
        return reaction_time + np.min(distances, axis=0) / max_speed

    attack_time = min_arrival(a_pos, a_vel)
    defense_time = min_arrival(d_pos, d_vel)
    return ControlSurface(xs, ys, expit((defense_time - attack_time) / time_scale), pitch_length, pitch_width)


def expected_threat(
    xs: ArrayLike,
    ys: ArrayLike,
    *,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    attack_direction: int = 1,
) -> FloatArray:
    """Sample xT at 1D physical grid centres, returning an (ny, nx) array.

    The original 12×8 bins are retained (no interpolation); coordinates at
    the pitch boundary map to the nearest edge bin. Off-pitch, NaN or infinite
    coordinates are rejected. Attack direction is either +1 or -1.
    """
    if attack_direction not in (-1, 1):
        raise ValueError("attack_direction must be +1 or -1")
    if not all(np.isfinite(v) and v > 0 for v in (pitch_length, pitch_width)):
        raise ValueError("Pitch dimensions must be finite and positive")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or y.size == 0:
        raise ValueError("xs and ys must be non-empty 1D arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.abs(x) > pitch_length / 2) or np.any(np.abs(y) > pitch_width / 2):
        raise ValueError("Coordinates must be finite and lie on the pitch")
    grid = _xt_grid()
    col = np.minimum(((attack_direction * x / pitch_length + 0.5) * grid.shape[1]).astype(int), grid.shape[1] - 1)
    row = np.minimum(((y / pitch_width + 0.5) * grid.shape[0]).astype(int), grid.shape[0] - 1)
    return grid[np.ix_(row, col)]


def summarize_control(surface: ControlSurface, *, attack_direction: int = 1) -> ControlMetrics:
    """Integrate control and control × xT over equal-area pitch cells.

    The second result has units xT·m² and is a spatial opportunity index,
    not additive xT from a ball action or the expected number of goals.
    """
    if surface.probability.shape != (len(surface.ys), len(surface.xs)):
        raise ValueError("Surface shape does not match grid coordinates")
    if not np.isfinite(surface.probability).all() or np.any((surface.probability < 0) | (surface.probability > 1)):
        raise ValueError("Control scores must be finite and between 0 and 1")
    grid = expected_threat(surface.xs, surface.ys, pitch_length=surface.pitch_length, pitch_width=surface.pitch_width, attack_direction=attack_direction)
    area = surface.pitch_length * surface.pitch_width / surface.probability.size
    return ControlMetrics(float(surface.probability.sum() * area), float((surface.probability * grid).sum() * area))
