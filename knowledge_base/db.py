"""SQLite audit log: trajectories, contexts, actions, rewards, policies."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DB_PATH = Path("knowledge_base/audit.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
  policy_id   TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  version     TEXT NOT NULL DEFAULT '0.1',
  hyperparams TEXT NOT NULL DEFAULT '{}',  -- JSON
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trajectories (
  trajectory_id    TEXT PRIMARY KEY,
  started_at       TIMESTAMP NOT NULL,
  project_id       TEXT NOT NULL,
  policy_id        TEXT NOT NULL REFERENCES policies(policy_id),
  drift_segment_id TEXT
);

CREATE TABLE IF NOT EXISTS contexts (
  context_id       TEXT PRIMARY KEY,
  trajectory_id    TEXT NOT NULL REFERENCES trajectories(trajectory_id),
  step             INTEGER NOT NULL,
  commit_sha       TEXT NOT NULL,
  feature_vector   TEXT NOT NULL,  -- JSON array of floats
  observed_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
  action_id      TEXT PRIMARY KEY,
  context_id     TEXT NOT NULL REFERENCES contexts(context_id),
  action         TEXT NOT NULL CHECK(action IN ('deploy', 'canary', 'block')),
  policy_id      TEXT NOT NULL REFERENCES policies(policy_id),
  propensity     REAL NOT NULL DEFAULT 1.0,
  decided_at     TIMESTAMP NOT NULL,
  explanation    TEXT NOT NULL DEFAULT '{}'  -- JSON
);

CREATE TABLE IF NOT EXISTS rewards (
  reward_id    TEXT PRIMARY KEY,
  action_id    TEXT NOT NULL REFERENCES actions(action_id),
  observed_at  TIMESTAMP NOT NULL,
  raw_outcome  TEXT NOT NULL CHECK(raw_outcome IN ('success', 'failure', 'censored', 'blocked')),
  cost         REAL NOT NULL,
  censored     BOOLEAN NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_trajectories_project
  ON trajectories(project_id);
CREATE INDEX IF NOT EXISTS idx_contexts_trajectory
  ON contexts(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_actions_context
  ON actions(context_id);
CREATE INDEX IF NOT EXISTS idx_rewards_action
  ON rewards(action_id);
"""


# ---------------------------------------------------------------------------
# Connection helpers (keep signature; update SQL string)
# ---------------------------------------------------------------------------

def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with dictionary-like rows."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the audit-log schema if it does not exist."""
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Per-table dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyRecord:
    policy_id: str
    name: str
    version: str = "0.1"
    hyperparams: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    started_at: str       # ISO-8601
    project_id: str
    policy_id: str
    drift_segment_id: str | None = None


@dataclass(frozen=True)
class ContextRecord:
    context_id: str
    trajectory_id: str
    step: int
    commit_sha: str
    feature_vector: list[float]
    observed_at: str      # ISO-8601


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    context_id: str
    action: str           # 'deploy' | 'canary' | 'block'
    policy_id: str
    propensity: float = 1.0
    decided_at: str = ""  # ISO-8601
    explanation: dict[str, Any] | None = None


@dataclass(frozen=True)
class RewardRecord:
    reward_id: str
    action_id: str
    observed_at: str      # ISO-8601
    raw_outcome: str      # 'success' | 'failure' | 'censored' | 'blocked'
    cost: float
    censored: bool = False


# ---------------------------------------------------------------------------
# Per-table insert helpers
# ---------------------------------------------------------------------------

def insert_policy(connection: sqlite3.Connection, record: PolicyRecord) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO policies(policy_id, name, version, hyperparams) VALUES (?,?,?,?)",
        (record.policy_id, record.name, record.version,
         json.dumps(record.hyperparams or {})),
    )
    connection.commit()


def insert_trajectory(connection: sqlite3.Connection, record: TrajectoryRecord) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO trajectories"
        "(trajectory_id, started_at, project_id, policy_id, drift_segment_id)"
        " VALUES (?,?,?,?,?)",
        (record.trajectory_id, record.started_at, record.project_id,
         record.policy_id, record.drift_segment_id),
    )
    connection.commit()


def insert_context(connection: sqlite3.Connection, record: ContextRecord) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO contexts"
        "(context_id, trajectory_id, step, commit_sha, feature_vector, observed_at)"
        " VALUES (?,?,?,?,?,?)",
        (record.context_id, record.trajectory_id, record.step,
         record.commit_sha, json.dumps(record.feature_vector), record.observed_at),
    )
    connection.commit()


def insert_action(connection: sqlite3.Connection, record: ActionRecord) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO actions"
        "(action_id, context_id, action, policy_id, propensity, decided_at, explanation)"
        " VALUES (?,?,?,?,?,?,?)",
        (record.action_id, record.context_id, record.action, record.policy_id,
         record.propensity, record.decided_at, json.dumps(record.explanation or {})),
    )
    connection.commit()


def insert_reward(connection: sqlite3.Connection, record: RewardRecord) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO rewards"
        "(reward_id, action_id, observed_at, raw_outcome, cost, censored)"
        " VALUES (?,?,?,?,?,?)",
        (record.reward_id, record.action_id, record.observed_at,
         record.raw_outcome, record.cost, int(record.censored)),
    )
    connection.commit()


# ---------------------------------------------------------------------------
# Per-table query helpers
# ---------------------------------------------------------------------------

def list_trajectories(
    connection: sqlite3.Connection,
    project_id: str | None = None,
    limit: int = 100,
) -> list[sqlite3.Row]:
    if project_id:
        return list(connection.execute(
            "SELECT * FROM trajectories WHERE project_id=? ORDER BY started_at DESC LIMIT ?",
            (project_id, limit),
        ))
    return list(connection.execute(
        "SELECT * FROM trajectories ORDER BY started_at DESC LIMIT ?", (limit,)
    ))


def list_rewards_for_trajectory(
    connection: sqlite3.Connection,
    trajectory_id: str,
) -> list[sqlite3.Row]:
    return list(connection.execute(
        """
        SELECT r.*
        FROM rewards r
        JOIN actions a ON r.action_id = a.action_id
        JOIN contexts c ON a.context_id = c.context_id
        WHERE c.trajectory_id = ?
        ORDER BY r.observed_at
        """,
        (trajectory_id,),
    ))


if __name__ == "__main__":
    initialize_database()
    print(f"Initialized {DEFAULT_DB_PATH}")
