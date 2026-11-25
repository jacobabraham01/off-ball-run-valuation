"""Behavioral checks for possession, coverage and direction-aware labels."""
import numpy as np
import pandas as pd
import pytest

from src.run_detector import detect_and_value_runs


def tracking(direction=1, scenario="behind"):
    """Six players, one known off-ball runner and 11 observed frames."""
    rows = []
    for frame in range(11):
        if scenario == "behind":
            runner_x, runner_y = direction * frame, 0.0
            defenders = [(3, -8), (8, 8), (20, 0)]
            ball_x, ball_y = -10, 0
        else:
            runner_x, runner_y = direction * (frame - 6), 12.0
            defenders = [(20, -8), (25, 8), (30, 0)]
            ball_x, ball_y = 0, 0
        positions = [(1, "home", runner_x, runner_y, 5),
                     (2, "home", -10, 0, 0), (3, "home", -20, -5, 0)]
        positions += [(i + 4, "away", direction * x, y, 0) for i, (x, y) in enumerate(defenders)]
        for player_id, team, x, y, speed in positions:
            rows.append(dict(period=1, frame=frame, timestamp=frame / 10,
                             player_id=player_id, team=team, x=x, y=y,
                             x_smooth=x, y_smooth=y, vx=direction * speed,
                             vy=0.0, speed=speed, is_detected=True,
                             possession_team="home", possession_player_id=2,
                             ball_x=direction * ball_x, ball_y=ball_y))
    return pd.DataFrame(rows)


def evaluate(data, direction=1, **kwargs):
    return detect_and_value_runs(data, {(1, "home"): direction}, min_players=3, **kwargs)


@pytest.mark.parametrize("direction", [1, -1])
def test_run_in_behind_is_direction_aware(direction):
    result = evaluate(tracking(direction).sample(frac=1, random_state=3), direction)
    assert len(result) == 1
    assert result.iloc[0].run_type == "run in behind"
    assert result.iloc[0].player_id == 1
    assert np.isfinite(result.iloc[0].space_created_m2)
    assert np.isfinite(result.iloc[0].xt_added_proxy)


def test_overlap_candidate_is_recognized():
    result = evaluate(tracking(scenario="overlap"))
    assert result.iloc[0].run_type == "overlap candidate"


@pytest.mark.parametrize("change", ["unknown_carrier", "runner_carrier", "possession_switch", "unknown_possession"])
def test_run_requires_known_opponent_free_possession_and_runner_off_ball(change):
    data = tracking()
    mask = data.frame == 5
    if change == "unknown_carrier":
        data.loc[mask, "possession_player_id"] = np.nan
    elif change == "runner_carrier":
        data.loc[mask, "possession_player_id"] = 1
    elif change == "possession_switch":
        data.loc[mask, "possession_team"] = "away"
    else:
        data.loc[mask, "possession_team"] = None
    assert evaluate(data).empty


def test_missing_runner_frame_cannot_be_bridged():
    data = tracking()
    data = data.loc[~((data.player_id == 1) & (data.frame == 5))]
    assert evaluate(data).empty


@pytest.mark.parametrize("failure", ["absent", "not_detected", "nonfinite"])
def test_endpoint_coverage_filters_low_visibility(failure):
    data = tracking()
    mask = (data.player_id == 4) & (data.frame == 10)
    if failure == "absent":
        data = data.loc[~mask]
    elif failure == "not_detected":
        data.loc[mask, "is_detected"] = False
    else:
        data["x_smooth"] = data["x_smooth"].astype(float)
        data.loc[mask, "x_smooth"] = np.inf
    assert evaluate(data, min_detected_fraction=0.9).empty


def test_missing_attack_direction_is_not_guessed():
    with pytest.raises(ValueError, match="attack direction"):
        detect_and_value_runs(tracking(), {}, min_players=3)
