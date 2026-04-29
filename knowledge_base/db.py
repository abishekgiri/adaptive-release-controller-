"""SQLite storage for deployment history records."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DB_PATH = Path("knowledge_base/deployments.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS deployments (
  id INTEGER PRIMARY KEY,
  commit_sha TEXT NOT NULL,
  files_changed INTEGER NOT NULL,
  lines_added INTEGER NOT NULL,
  lines_deleted INTEGER NOT NULL,
  test_passed BOOLEAN NOT NULL,
  ci_duration FLOAT NOT NULL,
  risk_score FLOAT NOT NULL,
  decision TEXT NOT NULL,
  outcome TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_deployments_commit_sha
ON deployments(commit_sha);

CREATE INDEX IF NOT EXISTS idx_deployments_outcome
ON deployments(outcome);

CREATE INDEX IF NOT EXISTS idx_deployments_created_at
ON deployments(created_at);
"""


@dataclass(frozen=True)
class DeploymentRecord:
    """A normalized deployment history row."""

    commit_sha: str
    files_changed: int
    lines_added: int
    lines_deleted: int
    test_passed: bool
    ci_duration: float
    risk_score: float
    decision: str
    outcome: str


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with dictionary-like rows."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the deployment history schema if it does not exist."""

    with connect(db_path) as connection:
        connection.executescript(SCHEMA)


def insert_deployment(
    connection: sqlite3.Connection,
    record: DeploymentRecord,
) -> int:
    """Insert one deployment record and return its row id."""

    cursor = connection.execute(
        """
        INSERT INTO deployments (
          commit_sha,
          files_changed,
          lines_added,
          lines_deleted,
          test_passed,
          ci_duration,
          risk_score,
          decision,
          outcome
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.commit_sha,
            record.files_changed,
            record.lines_added,
            record.lines_deleted,
            int(record.test_passed),
            record.ci_duration,
            record.risk_score,
            record.decision,
            record.outcome,
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def insert_many(
    connection: sqlite3.Connection,
    records: Iterable[DeploymentRecord],
) -> list[int]:
    """Insert multiple deployment records and return their row ids."""

    row_ids = []
    for record in records:
        row_ids.append(insert_deployment(connection, record))
    return row_ids


def list_deployments(
    connection: sqlite3.Connection,
    limit: int = 100,
) -> list[sqlite3.Row]:
    """Return recent deployment records."""

    return list(
        connection.execute(
            """
            SELECT *
            FROM deployments
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
    )


def get_deployment(
    connection: sqlite3.Connection,
    deployment_id: int,
) -> sqlite3.Row | None:
    """Return one deployment record by id."""

    return connection.execute(
        """
        SELECT *
        FROM deployments
        WHERE id = ?
        """,
        (deployment_id,),
    ).fetchone()


def count_deployments(connection: sqlite3.Connection) -> int:
    """Return the number of stored deployment records."""

    row = connection.execute("SELECT COUNT(*) AS total FROM deployments").fetchone()
    return int(row["total"])


def deployments_by_outcome(
    connection: sqlite3.Connection,
    outcome: str,
    limit: int = 100,
) -> list[sqlite3.Row]:
    """Return recent deployments matching an outcome."""

    return list(
        connection.execute(
            """
            SELECT *
            FROM deployments
            WHERE outcome = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (outcome, limit),
        )
    )


def failure_rate(connection: sqlite3.Connection) -> float:
    """Return the fraction of stored deployments marked as failure."""

    row = connection.execute(
        """
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END) AS failures
        FROM deployments
        """
    ).fetchone()
    total = int(row["total"])
    if total == 0:
        return 0.0
    return float(row["failures"] or 0) / total


if __name__ == "__main__":
    initialize_database()
    print(f"Initialized {DEFAULT_DB_PATH}")

