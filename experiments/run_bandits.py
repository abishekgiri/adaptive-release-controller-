"""Bandit experiment entrypoint for the first real result.

This delegates to ``run_baselines`` so LinUCB and the cost-sensitive delayed
bandit run over the exact same replay-evaluation path as the static baseline.
Thompson remains a TODO placeholder.
"""

from __future__ import annotations

from experiments.run_baselines import main


if __name__ == "__main__":
    main()
