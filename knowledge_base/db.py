"""SQLite storage for bandit experiment trajectories, actions, and rewards.

Schema (normalized, 5 tables):
    trajectories  — one row per policy run over one project
    contexts      — one row per context observation within a trajectory
    actions       — one row per decision taken by the policy
    rewards       — one row per matured reward delivered to the policy
    policies      — one row per registered policy configuration
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("knowledge_base/experiments.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
  policy_id   TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  version     TEXT NOT NULL DEFAULT '0.1.0',
  hyperparams_json TEXT NOT NULL DEFAULT '{}',
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trajectories (
  trajectory_id    TEXT PRIMARY KEY,
  started_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  project_id       TEXT NOT NULL,
  policy_id        TEXT NOT NULL REFERENCES policies(policy_id),
  drift_segment_id TEXT
);

CREATE TABLE IF NOT EXISTS contexts (
  context_id          TEXT PRIMARY KEY,
  trajectory_id       TEXT NOT NULL REFERENCES trajectories(trajectory_id),
  commit_sha          TEXT NOT NULL,
  feature_vector_json TEXT NOT NULL DEFAULT '{}',
  observed_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS actions (
  action_id        TEXT PRIMARY KEY,
  context_id       TEXT NOT NULL REFERENCES contexts(context_id),
  action           TEXT NOT NULL,
  policy_id        TEXT NOT NULL REFERENCES policies(policy_id),
  propensity       REAL NOT NULL,
  decided_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  explanation_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS rewards (
  reward_id   TEXT PRIMARY KEY,
  action_id   TEXT NOT NULL REFERENCES actions(action_id),
  observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  raw_outcome TEXT NOT NULL,
  cost        REAL NOT NULL,
  censored    BOOLEAN NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_trajectories_policy_id
ON trajectories(policy_id);

CREATE INDEX IF NOT EXISTS idx_contexts_trajectory_id
ON contexts(trajectory_id);

CREATE INDEX IF NOT EXISTS idx_actions_context_id
ON actions(context_id);

CREATE INDEX IF NOT EXISTS idx_rewards_action_id
ON rewards(action_id);
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with dictionary-like rows."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the experiment schema if it does not exist."""

    with connect(db_path) as connection:
        connection.executescript(SCHEMA)


if __name__ == "__main__":
    initialize_database()
    print(f"Initialized {DEFAULT_DB_PATH}")
