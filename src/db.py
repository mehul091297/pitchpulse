"""
SQLite connection helper for PitchPulse.

Keeps a single place that knows where the database lives, so ingest.py,
analysis.py, and the dashboard all read/write the same file consistently.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine

# Anchored to this file's own location (src/db.py -> src/ -> repo root),
# not to the process's current working directory. A plain relative path
# here would happen to work in Colab (cwd == repo root there) but isn't
# guaranteed elsewhere — the same class of bug that broke dashboard/app.py's
# `from src.analysis import ...` on Streamlit Community Cloud.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "processed" / "pitchpulse.db"


def get_engine(db_path: str | Path | None = None):
    """Return a SQLAlchemy engine pointed at the project's SQLite database.

    db_path defaults to the PITCHPULSE_DB_PATH env var, falling back to
    data/processed/pitchpulse.db (resolved from repo root) so this works
    with zero configuration during early development.
    """
    path = Path(db_path or os.getenv("PITCHPULSE_DB_PATH", DEFAULT_DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}")
