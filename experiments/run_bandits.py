"""Bandit experiment entrypoint for the first real result.

For Immediate Next Task 14 this delegates to ``run_baselines`` because LinUCB is
the only runnable bandit baseline. Thompson and the final contribution remain
TODO placeholders.
"""

from __future__ import annotations

from experiments.run_baselines import main


if __name__ == "__main__":
    main()
