"""Check physical symmetries, direction handling, and missing-player behavior."""
import numpy as np
import pytest

from src.spatial_model import expected_threat, pitch_control, summarize_control


def test_identical_teams_split_control_and_pitch_area() -> None:
    players = np.array([[0.0, 0.0], [10.0, -5.0]])
    control = pitch_control(players, players, nx=12, ny=8)
    np.testing.assert_allclose(control.probability, 0.5)
    assert summarize_control(control).controlled_area_m2 == pytest.approx(105 * 68 / 2)


def test_team_swap_complements_control() -> None:
    a = np.array([[-20.0, 0.0], [0.0, 5.0]])
    b = np.array([[15.0, -4.0]])
    first = pitch_control(a, b, nx=11, ny=7)
    second = pitch_control(b, a, nx=11, ny=7)
    np.testing.assert_allclose(first.probability + second.probability, 1.0)


def test_velocity_towards_target_increases_local_control() -> None:
    # At the grid centre the attacker moves towards the target, the defender
    # is stationary and an equal distance away.
    toward = pitch_control([[-10, 0]], [[10, 0]], [[3, 0]], nx=1, ny=1)
    still = pitch_control([[-10, 0]], [[10, 0]], nx=1, ny=1)
    assert toward.probability[0, 0] > still.probability[0, 0]


def test_xt_reverses_when_attack_direction_changes() -> None:
    x = np.array([-48.125, 0.0, 48.125])
    forward = expected_threat(x, [0.0], attack_direction=1)
    backward = expected_threat(x, [0.0], attack_direction=-1)
    assert forward[0, -1] > forward[0, 0]
    assert backward[0, 0] > backward[0, -1]
    assert forward[0, -1] == pytest.approx(backward[0, 0])


def test_missing_team_is_not_fabricated() -> None:
    with pytest.raises(ValueError, match="observed player"):
        pitch_control([[np.nan, np.nan]], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="on the pitch"):
        expected_threat([100.0], [0.0])


def test_xt_grid_loads_valid_local_json_and_rejects_bad_shape(tmp_path) -> None:
    import json
    from src.spatial_model import load_xt_grid

    path = tmp_path / 'grid.json'
    path.write_text(json.dumps([[0.03] * 12] * 8))
    grid = load_xt_grid(path)
    assert grid.shape == (8, 12)
    assert grid[0, 0] == pytest.approx(0.03)
    path.write_text(json.dumps([[0.03] * 11] * 8))
    with pytest.raises(ValueError, match='8×12'):
        load_xt_grid(path)
