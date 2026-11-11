"""Shared fixtures for the admin dashboard test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the parent directory is in sys.path for imports to work
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import pytest

from src.app import create_app


@pytest.fixture()
def app():
    app = create_app(testing=True)
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()
