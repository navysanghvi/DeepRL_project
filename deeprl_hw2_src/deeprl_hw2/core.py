"""Core classes."""
from collections import deque, namedtuple
import random
import numpy as np

# Used to store observed experience from an MDP. 
#Represents a standard (s, a, r, s', terminal) tuple.
sample = namedtuple('sample', 'state, action, reward, next_state, is_terminal')
        
class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        return self.data[(self.start + i) % self.maxlen]
    
    def append(self, value):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen  # Old values removed once memory fills up
        self.data[(self.start + self.length - 1) % self.maxlen] = value
    
class ReplayMemory(object):
    def __init__(self, max_size):                 
        self.window_length = window_length                          # Number of frames to stack together as a sequence
        self.max_size = max_size                                    # Maximum Replay memory size
        self.actions = RingBuffer(max_size)                         
        self.rewards = RingBuffer(max_size)
        self.terminals = RingBuffer(max_size)
        self.observations = RingBuffer(max_size)
        
    def sample(self, batch_size):  
        indexes = random.sample(xrange(1, len(self.observations)), batch_size) 
        sequence_batch = []
        
        for i in indexes:
            # State i-2 indicates if current state(i-1) is terminal or not 
            check_terminal = self.terminals[i - 2] if i>=2 else False
            while check_terminal: 
                i = random.sample(xrange(1,len(self.observations)), 1)[0]
                check_terminal = self.terminals[i - 2] if i >= 2 else False
            state = [self.observations[i - 1]]
            action = self.actions[i - 1]
            reward = self.rewards[i - 1]
            is_terminal = self.terminals[i - 1]
            next_state = [self.observations[i]]
            sequence_batch.append(sample(state=state, action=action, reward=reward,
                                          next_state=next_state, is_terminal=is_terminal))
        return sequence_batch

    def append(self, observation, action, reward, terminal):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
    
   



