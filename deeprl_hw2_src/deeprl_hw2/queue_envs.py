# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np


def categorical_sample(prob_n, rng):
    csprob_n = np.cumsum(np.asarray(prob_n))
    return (csprob_n > rng.rand()).argmax()


class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = np.prod(self.observation_space.high - self.observation_space.low + 1)
        self.nA = self.action_space.n
        self.p_add = [p1, p2, p3]
        self.isd1 = (1/6)*np.ones(6)
        self.isd2 = (1/6)*np.ones(6)
        self.isd3 = (1/6)*np.ones(6)

        self._seed()
        self._reset()
        self._populate_P()

    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        self.s = (1, categorical_sample(self.isd1, self.rng), 
            categorical_sample(self.isd2, self.rng), categorical_sample(self.isd3, self.rng))
        self.lastaction = None
        return self.s


    def _populate_P(self):
        limits = self.observation_space.high - self.observation_space.low + 1
        P = {}
        for i in range(limits[0]):
            for j in range(limits[1]):
                for k in range(limits[2]):
                    for l in range(limits[3]):
                        state = (self.observation_space.low[0]+i, 
                            self.observation_space.low[1]+j,
                            self.observation_space.low[2]+k,
                            self.observation_space.low[3]+l)
                        P[state] = {}
                        for a in range(self.nA):
                            P[state][a] = self.query_model(state, a)
        self.P = P


    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        transitions = self.query_model(self.s, action)
        prob, next_s, r, is_terminal = transitions[categorical_sample(
            (t[0] for t in transitions), self.rng)]

        next_s = np.asarray(next_s)
        for i in range(len(self.s) - 1):
            if next_s[i+1] < self.observation_space.high[i+1]:
                p = self.p_add[i]
                if(categorical_sample([p, 1-p], self.rng) == 0):
                    next_s[i+1] += 1

        self.s = tuple(next_s)
        self.lastaction = action
        return (next_s, r, is_terminal, {"prob": prob})

    def _render(self, mode='human', close=False):
        s = self.s
        for i in range(len(s)-1):
            print('Queue ' + str(i+1) + '  ->', end='\t')
            for j in range(self.observation_space.high[i+1]-self.observation_space.low[i+1] + 1):
                if(j < s[i+1]):
                    print(':---:', end='\t')
                else:
                    print(' ', end='\t')
            if(i+1 == s[0]):
                print('\b\b\b\b\b\b\b\b\b<< (Current Queue = ' + str(s[0]) + ')', end='\n')
            else:
                print(end='\n')
        print('\nLast Action: ' + self.get_action_name(self.lastaction))
        print('Current State: ', end=' ')
        print(s)
        print('\n')

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """

        state = np.asarray(state)
        if(action == 0):
            return [(1, (1, state[1], state[2], state[3]), 0, False)]
        if(action == 1):
            return [(1, (2, state[1], state[2], state[3]), 0, False)]
        if(action == 2):
            return [(1, (3, state[1], state[2], state[3]), 0, False)]
        if(state[state[0]] > self.observation_space.low[state[0]]):
            state[state[0]] -= 1
            reward = 1
        else:
            reward = 0

        return [(1, tuple(state), reward, False)]

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
