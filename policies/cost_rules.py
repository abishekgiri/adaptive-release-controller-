"""Simple cost-aware controls with no fitted contextual action-value model.

The Beta estimator learns one project-local failure probability from the same
matured CI proxy labels used by bandits. It is online learning, not a static
classifier, and assumes these labels are available after every simulated action.
"""
import math
import numpy as np
from data.schemas import Action, Outcome
from policies.base import Policy, FeatureEncoder
from rewards.cost_model import CostConfig


def expected_costs(p, cost_config):
    c = cost_config
    return {Action.DEPLOY: (1-p)*c.deploy_success+p*c.deploy_failure,
            Action.CANARY: (1-p)*c.canary_success+p*c.canary_failure,
            Action.BLOCK: (1-p)*c.block_safe+p*c.block_bad}


class ExpectedCostRule(Policy):
    """Fixed mapping from the available rolling failure rate to minimum cost."""
    def __init__(self, cost_config=CostConfig(), policy_id="cost_rule"):
        self.cost_config = cost_config
        self._policy_id = policy_id

    @property
    def policy_id(self):
        return self._policy_id

    def failure_probability(self, context):
        return context.recent_failure_rate

    def select_action(self, context):
        costs = expected_costs(self.failure_probability(context), self.cost_config)
        return min(costs, key=costs.get), 1.0

    def update(self, context, action, reward):
        pass

    def reset(self):
        pass


class BayesianRateRule(ExpectedCostRule):
    """Beta(1,1) posterior mean; one scalar rate, no contextual features."""
    def __init__(self, cost_config=CostConfig(), policy_id="bayesian_rate"):
        super().__init__(cost_config, policy_id)
        self.reset()

    def failure_probability(self, context):
        return self.failures / (self.failures + self.successes)

    def update(self, context, action, reward):
        if reward.censored or not math.isfinite(reward.cost):
            return
        if reward.outcome == Outcome.FAILURE:
            self.failures += 1
        elif reward.outcome == Outcome.SUCCESS:
            self.successes += 1

    def reset(self):
        self.failures = 1
        self.successes = 1


class ConstantActionPolicy(ExpectedCostRule):
    def __init__(self, action):
        super().__init__(policy_id=f"always_{action.value}")
        self.action = action

    def select_action(self, context):
        return self.action, 1.0


class BiasOnlyEncoder(FeatureEncoder):
    DIM = 1

    def encode(self, context):
        return np.ones(1)
