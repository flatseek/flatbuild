"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def torch():
    """Make torch importable from test bodies without try/except noise."""
    import torch  # noqa: F401

    return torch
