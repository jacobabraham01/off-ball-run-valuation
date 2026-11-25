"""Heuristic off-ball run classification and observed spatial change.

These labels and endpoint deltas are descriptive, not causal estimates. Every
run requires uninterrupted known team possession and a known ball carrier.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .kinematics import segment_runs
from .spatial_model import pitch_control, summarize_control


def _snapshot_metrics(snapshot: pd.DataFrame, team: str, direction: int,
                      length: float, width: float, min_players: int,
                      min_detected_fraction: float) -> tuple[float, float] | None:
    """Return spatial summaries, or None if either team's coverage is poor."""
    valid = snapshot.dropna(subset=["x_smooth", "y_smooth", "team"])
    valid = valid.loc[np.isfinite(valid[["x_smooth", "y_smooth"]].to_numpy(dtype=float)).all(axis=1)]
    attackers = valid.loc[valid.team == team]
    defenders = valid.loc[valid.team != team]
    if min(len(attackers), len(defenders)) < min_players:
        return None
    if "is_detected" in valid:
        for side in (attackers, defenders):
            if side.is_detected.fillna(False).astype(bool).mean() < min_detected_fraction:
                return None
    def arrays(side: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pos = side[["x_smooth", "y_smooth"]].to_numpy(dtype=float)
        vel = side[["vx", "vy"]].fillna(0).to_numpy(dtype=float)
        return pos, vel
    a, av = arrays(attackers)
    d, dv = arrays(defenders)
    surface = pitch_control(a, d, av, dv, pitch_length=length, pitch_width=width)
    metrics = summarize_control(surface, attack_direction=direction)
    return metrics.controlled_area_m2, metrics.threat_weighted_area_m2


def detect_and_value_runs(
    tracking: pd.DataFrame,
    attack_directions: Mapping[tuple[int, str], int],
    *, pitch_length: float = 105.0, pitch_width: float = 68.0,
    speed_threshold: float = 4.0, min_duration: float = 0.7,
    min_players: int = 7, min_detected_fraction: float = 0.5,
) -> pd.DataFrame:
    """Find eligible off-ball bouts and compare team control at their endpoints.

    ``attack_directions[(period, team)]`` must be +1 or -1, where team is
    ``home`` or ``away``.  This strict filter discards bouts on a possession
    change, unknown carrier or missing runner record; it never assumes that an
    unknown ball carrier makes a run off-ball. Deltas include all players'
    movement and do not attribute defenders' reactions to the runner.
    """
    required = {"frame", "period", "timestamp", "player_id", "team", "possession_team",
                "possession_player_id", "x_smooth", "y_smooth", "vx", "vy", "speed"}
    missing = required - set(tracking.columns)
    if missing:
        raise ValueError(f"tracking missing columns: {sorted(missing)}")
    if not 0 <= min_detected_fraction <= 1 or min_players < 1:
        raise ValueError("invalid coverage thresholds")
    bouts = segment_runs(tracking, speed_threshold=speed_threshold, min_duration=min_duration)
    columns = [*bouts.columns, "team", "run_type", "start_x", "start_y", "end_x", "end_y",
               "space_created_m2", "xt_weighted_space_added_m2", "xt_added_proxy"]
    if bouts.empty:
        return pd.DataFrame(columns=columns)
    by_frame = {(p, f): g for (p, f), g in tracking.groupby(["period", "frame"], sort=False)}
    by_player = {(p, pid): g.set_index("frame", drop=False).sort_index() for (p, pid), g in
                 tracking.groupby(["period", "player_id"], sort=False)}
    rows: list[dict] = []
    for bout in bouts.to_dict("records"):
        period, player = bout["period"], bout["player_id"]
        start, end = bout["start_frame"], bout["end_frame"]
        traj = by_player[(period, player)].loc[start:end]
        if len(traj) != end - start + 1 or traj.team.isna().any():
            continue
        team = traj.team.iloc[0]
        if team not in ("home", "away") or not (traj.team == team).all():
            continue
        if traj.possession_team.isna().any() or not (traj.possession_team == team).all():
            continue
        if traj.possession_player_id.isna().any() or (traj.possession_player_id == player).any():
            continue
        direction = attack_directions.get((int(period), team))
        if direction not in (-1, 1):
            raise ValueError(f"missing or invalid attack direction for {(period, team)}")
        first, last = by_frame.get((period, start)), by_frame.get((period, end))
        if first is None or last is None:
            continue
        m0 = _snapshot_metrics(first, team, direction, pitch_length, pitch_width, min_players, min_detected_fraction)
        m1 = _snapshot_metrics(last, team, direction, pitch_length, pitch_width, min_players, min_detected_fraction)
        if m0 is None or m1 is None:
            continue
        start_row, end_row = traj.iloc[0], traj.iloc[-1]
        x0, y0 = float(start_row.x_smooth), float(start_row.y_smooth)
        x1, y1 = float(end_row.x_smooth), float(end_row.y_smooth)
        defense = first.loc[(first.team != team) & first.team.notna()].dropna(subset=["x_smooth", "y_smooth"])
        kind = "other forward run" if direction * (x1 - x0) > 2 else "supporting run"
        # Deepest defender in attacking coordinates, excluding the most advanced
        # backstop (usually goalkeeper): second largest is a conservative line.
        if len(defense) >= 2:
            line = float(np.sort(direction * defense.x_smooth.to_numpy())[-2])
            if direction * x0 <= line and direction * x1 > line and direction * (x1 - x0) > 2:
                kind = "run in behind"
        bx0 = start_row.get("ball_x", np.nan)
        bx1 = end_row.get("ball_x", np.nan)
        by0 = start_row.get("ball_y", np.nan)
        if kind != "run in behind" and all(pd.notna(v) for v in (bx0, bx1, by0)):
            if abs(y0) > abs(float(by0)) + 3 and direction * (x0 - float(bx0)) < 0 <= direction * (x1 - float(bx1)):
                kind = "overlap candidate"
        if kind not in ("run in behind", "overlap candidate") and len(defense):
            near = defense.assign(distance=np.hypot(defense.x_smooth - x0, defense.y_smooth - y0)).sort_values("distance").iloc[0]
            later = last.loc[last.player_id == near.player_id]
            if len(later) and near.distance <= 4:
                defender = later.iloc[0]
                if pd.notna(defender.x_smooth) and np.hypot(defender.x_smooth - near.x_smooth, defender.y_smooth - near.y_smooth) >= 3 and np.hypot(defender.x_smooth - x1, defender.y_smooth - y1) <= 4:
                    kind = "defender-follow candidate"
        delta_space, delta_xt = m1[0] - m0[0], m1[1] - m0[1]
        rows.append({**bout, "team": team, "run_type": kind, "start_x": x0, "start_y": y0,
                     "end_x": x1, "end_y": y1, "space_created_m2": delta_space,
                     "xt_weighted_space_added_m2": delta_xt,
                     "xt_added_proxy": delta_xt / (pitch_length * pitch_width)})
    return pd.DataFrame(rows, columns=columns)
