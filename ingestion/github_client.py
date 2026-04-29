"""GitHub API client for commit and CI run ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GITHUB_API_BASE = "https://api.github.com"


@dataclass(frozen=True)
class DeploymentInput:
    """Raw GitHub inputs for one candidate deployment record."""

    commit: dict[str, Any]
    ci_run: dict[str, Any] | None


class GitHubClient:
    """Small GitHub REST client for repository history and Actions runs."""

    def __init__(
        self,
        owner: str,
        repo: str,
        token: str | None = None,
        api_base: str = GITHUB_API_BASE,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.token = token
        self.api_base = api_base.rstrip("/")

    def list_commits(
        self,
        per_page: int = 100,
        page: int = 1,
        branch: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return repository commits from newest to oldest."""

        params: dict[str, Any] = {"per_page": per_page, "page": page}
        if branch:
            params["sha"] = branch
        return self._request(f"/repos/{self.owner}/{self.repo}/commits", params)

    def get_commit(self, sha: str) -> dict[str, Any]:
        """Return full commit details including file stats."""

        return self._request(f"/repos/{self.owner}/{self.repo}/commits/{sha}")

    def list_workflow_runs(
        self,
        per_page: int = 100,
        page: int = 1,
        branch: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return GitHub Actions workflow runs."""

        params: dict[str, Any] = {"per_page": per_page, "page": page}
        if branch:
            params["branch"] = branch
        if status:
            params["status"] = status
        payload = self._request(
            f"/repos/{self.owner}/{self.repo}/actions/runs",
            params,
        )
        return payload.get("workflow_runs", [])

    def collect_deployment_inputs(
        self,
        limit: int = 100,
        branch: str | None = None,
    ) -> list[DeploymentInput]:
        """Combine commit details with the latest CI run for each commit SHA."""

        commits = self._collect_commits(limit=limit, branch=branch)
        runs = self._collect_workflow_runs(limit=limit, branch=branch)
        runs_by_sha = latest_runs_by_commit_sha(runs)

        deployment_inputs = []
        for commit_summary in commits[:limit]:
            sha = commit_summary["sha"]
            deployment_inputs.append(
                DeploymentInput(
                    commit=self.get_commit(sha),
                    ci_run=runs_by_sha.get(sha),
                )
            )
        return deployment_inputs

    def _collect_commits(
        self,
        limit: int,
        branch: str | None = None,
    ) -> list[dict[str, Any]]:
        """Collect enough commit summaries to satisfy the requested limit."""

        commits: list[dict[str, Any]] = []
        page = 1
        while len(commits) < limit:
            batch = self.list_commits(
                per_page=min(limit - len(commits), 100),
                page=page,
                branch=branch,
            )
            if not batch:
                break
            commits.extend(batch)
            page += 1
        return commits[:limit]

    def _collect_workflow_runs(
        self,
        limit: int,
        branch: str | None = None,
    ) -> list[dict[str, Any]]:
        """Collect enough workflow runs to match the requested dataset size."""

        runs: list[dict[str, Any]] = []
        page = 1
        while len(runs) < limit:
            batch = self.list_workflow_runs(
                per_page=min(limit - len(runs), 100),
                page=page,
                branch=branch,
            )
            if not batch:
                break
            runs.extend(batch)
            page += 1
        return runs[:limit]

    def _request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Send a GitHub API request and parse the JSON response."""

        query = f"?{urlencode(params)}" if params else ""
        request = Request(f"{self.api_base}{path}{query}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")

        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            message = error.read().decode("utf-8")
            raise RuntimeError(f"GitHub API request failed: {message}") from error


def latest_runs_by_commit_sha(
    workflow_runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index the newest workflow run by commit SHA."""

    indexed: dict[str, dict[str, Any]] = {}
    for run in workflow_runs:
        sha = run.get("head_sha")
        if not sha:
            continue
        existing = indexed.get(sha)
        if (
            existing is None
            or run.get("created_at", "") > existing.get("created_at", "")
        ):
            indexed[sha] = run
    return indexed
