"""Root conftest to handle Python path setup for nested test directories."""

import sys
from pathlib import Path

# Immediately add all necessary paths when this conftest is imported
root_dir = Path(__file__).resolve().parent

# Add fixture directories to path
for fixture_dir in root_dir.glob("fixtures/app-*"):
    if fixture_dir.is_dir() and (fixture_dir / "src").exists():
        fixture_path = str(fixture_dir)
        if fixture_path not in sys.path:
            sys.path.insert(0, fixture_path)

# Add multi_agent workspace directories to path
for workspace_dir in root_dir.glob("multi_agent/workspaces/*/app-*"):
    if workspace_dir.is_dir() and (workspace_dir / "src").exists():
        workspace_path = str(workspace_dir)
        if workspace_path not in sys.path:
            sys.path.insert(0, workspace_path)


def pytest_configure(config):
    """Configure pytest to handle imports from nested test directories."""
    # The root directory
    root_dir = Path(__file__).resolve().parent

    # Add root to path for vendor tests
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
