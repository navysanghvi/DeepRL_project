"""RL Policy classes.

We have provided you with a base policy class, some example
implementations and some unimplemented classes that should be useful
in your code.
"""
import numpy as np


class Policy(object):

    def track_agent(self, agent):
        self.agent = agent
    
    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        return {}


class UniformRandomPolicy(Policy):      # Chooses an action with uniform random probability

    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self, **kwargs):
        return np.random.randint(0, self.num_actions)

    def get_config(self):  # noqa: D102
        return {'num_actions': self.num_actions}


class GreedyPolicy(Policy):             # Always returns best action according to Q-values

    def select_action(self, q_values, **kwargs):  # noqa: D102
        return np.argmax(q_values)


class GreedyEpsilonPolicy(Policy):       # Selects greedy action or with some probability epsilon a random action
    def __init__(self, epsilon=0.05):
        super(GreedyEpsilonPolicy, self).__init__()
        self.eps = epsilon

    def select_action(self, q_values):
        num_actions = q_values.shape[0]
        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, num_actions-1)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        config = super(GreedyEpsilonPolicy, self).get_config()
        config['epsilon'] = self.epsilon
        return config

class LinearDecayGreedyEpsilonPolicy(Policy):       # Like GreedyEpsilonPolicy but epsilon decays from a start value
                                                    # to an end value over num_steps.
        
    def __init__(self, policy, attr_name, start_value, end_value, num_steps):
        super(LinearDecayGreedyEpsilonPolicy, self).__init__()
        self.policy = policy
        self.attr_name = attr_name
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps

    def select_action(self, **kwargs):
        step_val = -float(self.start_value - self.end_value) / float(self.num_steps)
        start_val = float(self.start_value)
        value = max(self.end_value, step_val * float(self.agent.step) + start_val)
        setattr(self.policy, self.attr_name, value)
        return self.policy.select_action(**kwargs)
    
    def get_config(self):
        config = super(LinearDecayGreedyEpsilonPolicy, self).get_config()
        config['attr_name'] = self.attr_name
        config['start_value'] = self.start_value
        config['end_value'] = self.end_value
        config['num_steps'] = self.num_steps
        config['policy'] = get_object_config(self.policy)
        return config
