"""Gap-aware player kinematics in metres and seconds.

Tracking is expected in long form: one row per player and frame, with integer
``frame``, ``period``, ``player_id``, seconds-valued ``timestamp``, and metric
``x`` / ``y``.  Missing positions are never extrapolated.  Derivatives are
computed separately for each continuous visible/interpolated trajectory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

_GROUPS = ["period", "player_id"]
_REQUIRED = {*_GROUPS, "frame", "timestamp", "x", "y"}


def _validate(tracking: pd.DataFrame) -> None:
    missing = _REQUIRED.difference(tracking.columns)
    if missing:
        raise ValueError(f"tracking is missing required columns: {sorted(missing)}")
    if tracking.duplicated([*_GROUPS, "frame"]).any():
        raise ValueError("each player may have at most one record per period and frame")
    if tracking[[*_GROUPS, "frame"]].isna().any().any():
        raise ValueError("period, player_id and frame cannot be null")
    frames = pd.to_numeric(tracking["frame"], errors="coerce")
    if frames.isna().any() or not np.equal(frames, np.floor(frames)).all():
        raise ValueError("frame values must be integers")


def interpolate_short_gaps(tracking: pd.DataFrame, max_gap_frames: int = 5) -> pd.DataFrame:
    """Fill bounded gaps of at most ``max_gap_frames``; do not extrapolate.

    Absent rows and rows with missing coordinates both count as gaps. Longer
    gaps remain wholly missing, preventing invented paths across off-camera
    intervals. ``is_interpolated`` identifies positions filled by this method.
    Frame indices must be sequential within a period; timestamps are seconds.
    """
    _validate(tracking)
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")
    if tracking.empty:
        result = tracking.copy()
        result["is_interpolated"] = pd.Series(dtype=bool)
        return result

    pieces: list[pd.DataFrame] = []
    for (period, player_id), group in tracking.groupby(_GROUPS, sort=False):
        group = group.sort_values("frame").copy()
        group["frame"] = group["frame"].astype(int)
        # Reindex so completely absent player records can also be interpolated.
        full = group.set_index("frame").reindex(
            range(int(group["frame"].min()), int(group["frame"].max()) + 1)
        )
        full.index.name = "frame"
        full["period"] = period
        full["player_id"] = player_id
        for col in ("team", "team_id"):
            if col in full:
                full[col] = full[col].ffill().bfill()
        if "is_detected" in full:
            full["is_detected"] = full["is_detected"].astype("boolean").fillna(False).astype(bool)
        full["timestamp"] = pd.to_numeric(full["timestamp"], errors="coerce").interpolate(limit_area="inside")
        for col in ("x", "y"):
            full[col] = pd.to_numeric(full[col], errors="coerce")
        valid = full[["x", "y"]].notna().all(axis=1)
        # Treat a missing coordinate as a missing position, not half a position.
        full.loc[~valid, ["x", "y"]] = np.nan
        gap = ~valid
        gap_sizes = gap.groupby((gap != gap.shift()).cumsum()).transform("sum")
        bounded = valid.cumsum().gt(0) & valid.iloc[::-1].cumsum().iloc[::-1].gt(0)
        eligible = gap & bounded & (gap_sizes <= max_gap_frames)
        candidate = full[["x", "y"]].interpolate(limit_area="inside")
        full.loc[eligible, ["x", "y"]] = candidate.loc[eligible, ["x", "y"]]
        full["is_interpolated"] = eligible
        pieces.append(full.reset_index())
    return pd.concat(pieces, ignore_index=True)


def _continuous_chunks(group: pd.DataFrame) -> list[np.ndarray]:
    """Indices of runs with valid position, increasing time and adjacent frames."""
    x = group["x"].to_numpy(dtype=float)
    y = group["y"].to_numpy(dtype=float)
    t = group["timestamp"].to_numpy(dtype=float)
    frames = group["frame"].to_numpy(dtype=int)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    split = np.ones(len(group), dtype=bool)
    if len(group) > 1:
        split[1:] = (~valid[1:] | ~valid[:-1] | (np.diff(frames) != 1) | (np.diff(t) <= 0))
    chunk_ids = np.cumsum(split)
    return [np.flatnonzero(valid & (chunk_ids == i)) for i in np.unique(chunk_ids[valid])]


def add_kinematics(
    tracking: pd.DataFrame,
    *,
    max_gap_frames: int = 5,
    smoothing_window: int = 7,
    polyorder: int = 2,
) -> pd.DataFrame:
    """Add smoothed positions, velocity (m/s) and acceleration (m/s²).

    The Savitzky–Golay filter is applied to positions, then derivatives use
    actual timestamps. A shorter odd window is used for short trajectories;
    those too short for the polynomial are left unsmoothed. Isolated samples
    have undefined derivatives. Inputs must use metres and seconds.
    """
    if smoothing_window < 1 or smoothing_window % 2 != 1:
        raise ValueError("smoothing_window must be a positive odd integer")
    if polyorder < 0:
        raise ValueError("polyorder must be non-negative")
    result = interpolate_short_gaps(tracking, max_gap_frames=max_gap_frames)
    columns = ["x_smooth", "y_smooth", "vx", "vy", "speed", "ax", "ay", "acceleration"]
    for col in columns:
        result[col] = np.nan
    if result.empty:
        return result
    for _, group in result.groupby(_GROUPS, sort=False):
        group = group.sort_values("frame")
        for indices in _continuous_chunks(group):
            subset = group.iloc[indices]
            row_indices = subset.index
            t = subset["timestamp"].to_numpy(dtype=float)
            x = subset["x"].to_numpy(dtype=float)
            y = subset["y"].to_numpy(dtype=float)
            window = min(smoothing_window, len(subset) if len(subset) % 2 else len(subset) - 1)
            if window > polyorder and window >= 3:
                x = savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp")
                y = savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")
            result.loc[row_indices, ["x_smooth", "y_smooth"]] = np.column_stack([x, y])
            if len(subset) < 2:
                continue
            vx = np.gradient(x, t)
            vy = np.gradient(y, t)
            result.loc[row_indices, ["vx", "vy", "speed"]] = np.column_stack([vx, vy, np.hypot(vx, vy)])
            if len(subset) >= 3:
                ax = np.gradient(vx, t)
                ay = np.gradient(vy, t)
                result.loc[row_indices, ["ax", "ay", "acceleration"]] = np.column_stack([ax, ay, np.hypot(ax, ay)])
    return result


def segment_runs(
    tracking: pd.DataFrame,
    *,
    speed_threshold: float = 4.0,
    sprint_threshold: float = 7.0,
    min_duration: float = 0.7,
    max_below_frames: int = 1,
) -> pd.DataFrame:
    """Return sustained high-speed bouts, independent of possession.

    Values are configurable illustrative thresholds, not calibrated labels.
    Short finite dips in speed can be bridged; missing tracking is never bridged.
    Duration is elapsed seconds between first and last sample. Distance is a
    trapezoidal integral of speed over that interval.
    """
    _validate(tracking)
    if "speed" not in tracking:
        raise ValueError("run add_kinematics first to provide speed")
    if speed_threshold < 0 or sprint_threshold < speed_threshold or min_duration < 0 or max_below_frames < 0:
        raise ValueError("invalid segmentation thresholds")
    output_columns = ["period", "player_id", "start_frame", "end_frame", "start_time", "end_time", "duration_s", "distance_m", "peak_speed_mps", "intensity"]
    bouts: list[dict[str, object]] = []
    for (period, player_id), group in tracking.groupby(_GROUPS, sort=False):
        group = group.sort_values("frame")
        for idx in _continuous_chunks(group):
            chunk = group.iloc[idx]
            speed = chunk["speed"].to_numpy(dtype=float)
            active = np.isfinite(speed) & (speed >= speed_threshold)
            # Fill only short dips bounded by active samples and finite speeds.
            inactive = ~active
            blocks = pd.Series(inactive).groupby(pd.Series(inactive).ne(pd.Series(inactive).shift()).cumsum()).transform("sum").to_numpy()
            left = np.cumsum(active) > 0
            right = np.cumsum(active[::-1])[::-1] > 0
            active |= inactive & np.isfinite(speed) & left & right & (blocks <= max_below_frames)
            changes = np.diff(np.r_[False, active, False].astype(int))
            for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                bout = chunk.iloc[start:end]
                times = bout["timestamp"].to_numpy(dtype=float)
                duration = float(times[-1] - times[0])
                if duration < min_duration:
                    continue
                speeds = bout["speed"].to_numpy(dtype=float)
                peak = float(np.max(speeds))
                bouts.append({"period": period, "player_id": player_id, "start_frame": int(bout["frame"].iloc[0]), "end_frame": int(bout["frame"].iloc[-1]), "start_time": float(times[0]), "end_time": float(times[-1]), "duration_s": duration, "distance_m": float(trapezoid(speeds, times)), "peak_speed_mps": peak, "intensity": "sprint" if peak >= sprint_threshold else "run"})
    return pd.DataFrame(bouts, columns=output_columns)
