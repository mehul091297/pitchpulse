"""
SQLite connection helper for PitchPulse.

Keeps a single place that knows where the database lives, so ingest.py,
analysis.py, and the dashboard all read/write the same file consistently.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine

DEFAULT_DB_PATH = Path("data/processed/pitchpulse.db")


def get_engine(db_path: str | Path | None = None):
    """Return a SQLAlchemy engine pointed at the project's SQLite database.

    db_path defaults to the PITCHPULSE_DB_PATH env var, falling back to
    data/processed/pitchpulse.db so this works with zero configuration
    during early development.
    """
    path = Path(db_path or os.getenv("PITCHPULSE_DB_PATH", DEFAULT_DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}")
