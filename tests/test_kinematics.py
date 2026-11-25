"""Tests for physical units, short-gap treatment and run segmentation."""
import numpy as np
import pandas as pd
import pytest

from src.kinematics import add_kinematics, interpolate_short_gaps, segment_runs


def make_track(frames, x, y=None):
    frames = np.asarray(frames)
    return pd.DataFrame({
        "period": 1, "player_id": 42, "frame": frames,
        "timestamp": frames / 10.0, "x": x,
        "y": np.zeros(len(frames)) if y is None else y,
        "team": "home", "is_detected": True,
    })


def test_constant_velocity_is_metres_per_second():
    frames = np.arange(30)
    result = add_kinematics(make_track(frames, frames * 0.4))
    np.testing.assert_allclose(result["vx"], 4.0, atol=1e-10)
    np.testing.assert_allclose(result["vy"], 0.0, atol=1e-10)
    np.testing.assert_allclose(result["speed"], 4.0, atol=1e-10)
    np.testing.assert_allclose(result["acceleration"], 0.0, atol=1e-9)


def test_short_absent_and_null_frames_are_interpolated():
    track = make_track([0, 1, 3, 4, 5], [0, 1, 3, np.nan, 5])
    result = interpolate_short_gaps(track, max_gap_frames=1)
    np.testing.assert_allclose(result["x"], np.arange(6))
    assert result.loc[result.is_interpolated, "frame"].tolist() == [2, 4]
    assert not result.loc[result.frame == 2, "is_detected"].iloc[0]
    assert result.loc[result.frame == 2, "timestamp"].iloc[0] == pytest.approx(0.2)


def test_long_gaps_remain_missing_and_are_not_bridged():
    track = make_track([0, 1, 5, 6], [0, 1, 5, 6])
    result = add_kinematics(track, max_gap_frames=2)
    assert result.loc[result.frame.isin([2, 3, 4]), "x"].isna().all()
    assert result.loc[result.frame.isin([2, 3, 4]), "speed"].isna().all()
    assert not result.is_interpolated.any()
    assert len(segment_runs(result, speed_threshold=1, min_duration=0)) == 2


def test_interpolation_does_not_extrapolate_at_ends():
    result = interpolate_short_gaps(make_track(range(4), [np.nan, 1, 2, np.nan]))
    assert pd.isna(result.x.iloc[0]) and pd.isna(result.x.iloc[-1])
    assert not result.is_interpolated.any()


def test_periods_are_never_connected():
    first = make_track([0, 1], [0, 1])
    second = make_track([2, 3], [100, 101]).assign(period=2)
    result = add_kinematics(pd.concat([first, second], ignore_index=True))
    np.testing.assert_allclose(result["speed"], 10)


def test_segmentation_uses_elapsed_time_and_bridges_short_finite_dip():
    track = make_track(range(10), np.arange(10) * 0.5)
    track["speed"] = [0, 5, 5, 3, 5, 5, 5, 5, 0, 0]
    result = segment_runs(track, speed_threshold=4, min_duration=0.5, max_below_frames=1)
    assert len(result) == 1
    assert (result.iloc[0].start_frame, result.iloc[0].end_frame) == (1, 7)
    assert result.iloc[0].duration_s == pytest.approx(0.6)
    assert result.iloc[0].distance_m == pytest.approx(2.8)


def test_duplicate_player_frames_are_rejected():
    with pytest.raises(ValueError, match="at most one"):
        add_kinematics(make_track([0, 0], [0, 1]))
