"""Load SkillCorner open broadcast tracking as a tidy, centre-origin table.

Source: https://github.com/SkillCorner/opendata. The provider's JSONL has one
record per 10 Hz video frame. Missing/off-camera observations are never filled
in here; ``is_detected=False`` denotes a provider-extrapolated observation,
not a directly observed position. See the repository README for caveats.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

DEFAULT_MATCH_ID = 1874553
RAW_ROOT = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"
# Tracking files are Git LFS objects. The ordinary raw URL can yield a pointer.
LFS_ROOT = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"
COLUMNS = [
    "frame", "period", "timestamp", "player_id", "x", "y", "team_id", "team",
    "is_detected", "ball_x", "ball_y", "ball_is_detected", "possession_team",
    "possession_player_id",
]


def _download(url: str, destination: Path) -> None:
    """Stream to an atomic cache file; leave no partial file on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        request = Request(url, headers={"User-Agent": "off-ball-space-valuation/1.0"})
        with urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, delete=False
        ) as out:
            temporary = out.name
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if Path(temporary).stat().st_size == 0:
            raise ValueError(f"Empty download from {url}")
        with open(temporary, "rb") as check:
            if check.read(40).startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError(f"Got a Git LFS pointer instead of tracking data: {url}")
        os.replace(temporary, destination)
        temporary = None
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def fetch_match(match_id: int = DEFAULT_MATCH_ID, cache_dir: str | Path = "data") -> tuple[Path, Path]:
    """Download the match metadata and extrapolated JSONL once and return paths.

    Git LFS tracking files can be large. Delete cached files to refresh them.
    Only numeric match IDs are accepted. The upstream data is subject to its own
    license and attribution; it is not bundled with this project.
    """
    if isinstance(match_id, bool) or not str(match_id).isdigit():
        raise ValueError("match_id must be a numeric SkillCorner match ID")
    match_id = int(match_id)
    directory = Path(cache_dir) / str(match_id)
    metadata = directory / f"{match_id}_match.json"
    tracking = directory / f"{match_id}_tracking_extrapolated.jsonl"
    if not metadata.exists():
        _download(f"{RAW_ROOT}/{match_id}/{metadata.name}", metadata)
    if not tracking.exists():
        _download(f"{LFS_ROOT}/{match_id}/{tracking.name}", tracking)
    return metadata, tracking


def timestamp_seconds(value: Any) -> float | None:
    """Convert HH:MM:SS.sss (or numeric seconds) to match-clock seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid match timestamp: {value!r}")
    try:
        hours, minutes, seconds = map(float, parts)
    except ValueError as exc:
        raise ValueError(f"Invalid match timestamp: {value!r}") from exc
    return hours * 3600 + minutes * 60 + seconds


def parse_tracking(
    frames: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> pd.DataFrame:
    """Flatten provider frames to one row per *available* player observation.

    Bounds are inclusive video-frame IDs, not match seconds. No absent players
    or off-camera frames are invented, and null coordinates stay null. The
    metadata determines team membership; unknown players retain a null team.
    """
    if start_frame is not None and end_frame is not None and start_frame > end_frame:
        raise ValueError("start_frame must be <= end_frame")
    home = (metadata.get("home_team") or {}).get("id")
    away = (metadata.get("away_team") or {}).get("id")
    players = {p.get("id"): p for p in metadata.get("players", [])}
    rows: list[dict[str, Any]] = []
    for record in frames:
        frame = record.get("frame")
        if frame is None or record.get("period") is None or record.get("timestamp") is None:
            continue
        if start_frame is not None and frame < start_frame:
            continue
        if end_frame is not None and frame > end_frame:
            continue
        possession = record.get("possession") or {}
        # The public JSONL uses "home team" / "away team"; normalize labels.
        group = possession.get("group")
        possession_team = {"home team": "home", "away team": "away", "home": "home", "away": "away"}.get(group)
        ball = record.get("ball_data") or {}
        for player in record.get("player_data") or []:
            player_id = player.get("player_id")
            if player_id is None:
                continue
            team_id = players.get(player_id, {}).get("team_id")
            team = "home" if team_id is not None and team_id == home else "away" if team_id is not None and team_id == away else None
            rows.append({
                "frame": frame, "period": record["period"],
                "timestamp": timestamp_seconds(record["timestamp"]),
                "player_id": player_id, "x": player.get("x"), "y": player.get("y"),
                "team_id": team_id, "team": team, "is_detected": player.get("is_detected"),
                "ball_x": ball.get("x"), "ball_y": ball.get("y"),
                "ball_is_detected": ball.get("is_detected"),
                "possession_team": possession_team,
                "possession_player_id": possession.get("player_id"),
            })
    return pd.DataFrame.from_records(rows, columns=COLUMNS)


def load_local_match(
    metadata_path: str | Path,
    tracking_path: str | Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read local official match files, optionally restricting inclusive frames."""
    with Path(metadata_path).open(encoding="utf-8") as file:
        metadata = json.load(file)
    with Path(tracking_path).open(encoding="utf-8") as file:
        # Iteration avoids holding the raw JSONL and the resulting table at once.
        frames = (json.loads(line) for line in file if line.strip())
        tracking = parse_tracking(frames, metadata, start_frame=start_frame, end_frame=end_frame)
    return tracking, metadata


def load_match(
    match_id: int = DEFAULT_MATCH_ID,
    cache_dir: str | Path = "data",
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch (or reuse) the open match and return tracking plus raw metadata."""
    metadata_path, tracking_path = fetch_match(match_id, cache_dir)
    return load_local_match(metadata_path, tracking_path, start_frame=start_frame, end_frame=end_frame)
