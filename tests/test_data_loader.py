"""Parser contract checks using a tiny SkillCorner-shaped fixture."""
from src.data_loader import parse_tracking, timestamp_seconds


def test_loader_preserves_extrapolation_and_missing_data():
    metadata = {
        "home_team": {"id": 10}, "away_team": {"id": 20},
        "players": [{"id": 1, "team_id": 10}, {"id": 2, "team_id": 20}],
    }
    frames = [
        {"frame": 10, "period": 1, "timestamp": "00:00:01.00",
         "ball_data": {"x": None, "y": None, "is_detected": False},
         "possession": {"player_id": 1, "group": "home team"},
         "player_data": [{"player_id": 1, "x": 0.0, "y": 3.0, "is_detected": False},
                         {"player_id": 2, "x": None, "y": None, "is_detected": None}]},
        {"frame": 11, "period": 1, "timestamp": "00:00:01.10", "player_data": []},
        {"frame": 12, "period": None, "timestamp": None, "player_data": []},
    ]
    result = parse_tracking(frames, metadata)
    assert len(result) == 2
    assert result.loc[0, "timestamp"] == 1.0
    assert result.loc[0, "team"] == "home"
    assert result.loc[0, "possession_team"] == "home"
    assert result.loc[0, "is_detected"] is False
    assert result.loc[1, "team"] == "away"
    assert result.loc[1, "x"] != result.loc[1, "x"]  # NaN remains missing
    assert parse_tracking(frames, metadata, start_frame=11).empty


def test_timestamp_handles_hours_and_tenths():
    assert timestamp_seconds("01:02:03.40") == 3723.4
    assert timestamp_seconds(None) is None
