"""Broadcast-style freeze frames and short tracking animations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from mplsoccer import Pitch

from .spatial_model import ControlSurface


def plot_frame(frame: pd.DataFrame, *, pitch_length: float = 105, pitch_width: float = 68,
               control: ControlSurface | None = None, trajectory: pd.DataFrame | None = None,
               title: str = "Tracking freeze frame") -> tuple[plt.Figure, plt.Axes]:
    """Plot centered coordinates on a metric pitch; optional control and run."""
    pitch = Pitch(pitch_type="custom", pitch_length=pitch_length, pitch_width=pitch_width,
                  pitch_color="#163d30", line_color="#edf5ef", linewidth=1.1)
    fig, ax = pitch.draw(figsize=(12, 7), constrained_layout=True)
    fig.set_facecolor("#102b24")
    if control is not None:
        ax.imshow(control.probability, origin="lower", extent=(0, pitch_length, 0, pitch_width),
                  cmap="coolwarm", vmin=0, vmax=1, alpha=0.58, aspect="auto", zorder=1)
    colors = {"home": "#57c4ff", "away": "#ffbd67"}
    for team, color in colors.items():
        side = frame.loc[frame.team == team]
        if side.empty:
            continue
        xcol = "x_smooth" if "x_smooth" in side else "x"
        ycol = "y_smooth" if "y_smooth" in side else "y"
        ax.scatter(side[xcol] + pitch_length / 2, side[ycol] + pitch_width / 2,
                   s=115, c=color, edgecolors="white", linewidths=0.6, zorder=4, label=team.title())
    if len(frame) and {"ball_x", "ball_y"} <= set(frame):
        ball = frame[["ball_x", "ball_y"]].dropna()
        if len(ball):
            ax.scatter(ball.iloc[0, 0] + pitch_length / 2, ball.iloc[0, 1] + pitch_width / 2,
                       c="white", edgecolors="#222222", s=38, zorder=6, label="Ball")
    if trajectory is not None and len(trajectory):
        xcol = "x_smooth" if "x_smooth" in trajectory else "x"
        ycol = "y_smooth" if "y_smooth" in trajectory else "y"
        ax.plot(trajectory[xcol] + pitch_length / 2, trajectory[ycol] + pitch_width / 2,
                color="#ffffff", linewidth=3, zorder=5, label="Runner")
        ax.scatter([trajectory[xcol].iloc[-1] + pitch_length / 2],
                   [trajectory[ycol].iloc[-1] + pitch_width / 2], s=180,
                   facecolors="none", edgecolors="white", linewidths=2, zorder=7)
    ax.set_title(title, color="white", loc="left", fontsize=16, fontweight="bold", pad=14)
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)
    for text in legend.get_texts():
        text.set_color("white")
    return fig, ax


def animate_tracking(tracking: pd.DataFrame, output_path: str | Path, *, fps: int = 10,
                     pitch_length: float = 105, pitch_width: float = 68,
                     step: int = 1, title: str = "Tracking") -> Path:
    """Write a GIF of one continuous period; step subsamples at unchanged speed."""
    if tracking.empty or tracking.period.nunique() != 1:
        raise ValueError("provide non-empty tracking from exactly one period")
    if fps <= 0 or step <= 0:
        raise ValueError("fps and step must be positive")
    frames = sorted(tracking.frame.unique())[::step]
    if fps / step <= 0:
        raise ValueError("invalid playback rate")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    first = tracking.loc[tracking.frame == frames[0]]
    fig, ax = plot_frame(first, pitch_length=pitch_length, pitch_width=pitch_width, title=title)
    def update(index: int) -> list:
        ax.clear()
        Pitch(pitch_type="custom", pitch_length=pitch_length, pitch_width=pitch_width,
              pitch_color="#163d30", line_color="#edf5ef").draw(ax=ax)
        current = tracking.loc[tracking.frame == frames[index]]
        for team, color in (("home", "#57c4ff"), ("away", "#ffbd67")):
            side = current[current.team == team]
            xc, yc = ("x_smooth", "y_smooth") if "x_smooth" in side else ("x", "y")
            ax.scatter(side[xc] + pitch_length / 2, side[yc] + pitch_width / 2,
                       s=90, c=color, edgecolors="white", linewidths=0.5, zorder=4)
        if {"ball_x", "ball_y"} <= set(current):
            ball = current[["ball_x", "ball_y"]].dropna()
            if not ball.empty:
                ax.scatter(ball.iloc[0, 0] + pitch_length / 2, ball.iloc[0, 1] + pitch_width / 2,
                           s=35, c="white", edgecolors="black", zorder=5)
        ax.set_title(f"{title}  |  frame {frames[index]}", color="white", loc="left", pad=14)
        return []
    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 * step / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps / step))
    plt.close(fig)
    return output
