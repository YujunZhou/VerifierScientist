"""Root conftest to add project root to Python path."""

import sys
from pathlib import Path

# Add project root to sys.path so imports like 'from src.*' work
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
