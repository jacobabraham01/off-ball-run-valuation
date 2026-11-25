"""Synthetic grid keeps unit tests independent of network and third-party data."""
import numpy as np
import pytest
import src.spatial_model as spatial


@pytest.fixture(autouse=True)
def synthetic_xt_grid(monkeypatch):
    monkeypatch.setattr(spatial, "_XT_GRID", np.tile(np.linspace(0.001, 0.1, 12), (8, 1)))
