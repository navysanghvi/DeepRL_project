#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import sys
sys.path.append('deeprl_hw2_src')

import argparse
import os
import random
from PIL import Image
import numpy as np
import gym
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)

from keras.models import Sequential
from keras.optimizers import Adam
import keras.backend as K

from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy
from deeprl_hw2.core import ReplayMemory
from deeprl_hw2.preprocessors import AtariPreprocessor, HistoryPreprocessor
from deeprl_hw2.callbacks import Log_File, Save_weights


def create_model(window, new_size, num_actions, is_linear=False,
                 model_name='q_network'):  # noqa: D103
    
    # Create layers according to linear or deep network requirements
    input_length = (window,) + new_size
    model = Sequential(name = model_name)
    model.add(Permute((2, 3, 1), input_shape = input_length))
    if not is_linear:
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))
    model.add(Flatten())
    if not is_linear:
        model.add(Dense(512))
        model.add(Activation('relu'))
    model.add(Dense(num_actions))
    if not is_linear:
        model.add(Activation('linear'))
    return model


def main():  # noqa: D103
    
    # Arguments
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    args = parser.parse_args()

    # Downsampled frame size
    new_size = (84,84)

    # Number of frames to input to the network as a Sequence
    window_size = 4

    # Linear network or not; Double network or not; Dueling network or not; no replay?
    is_linear = False; is_double = False; is_dueling = False; no_replay = False

    # Create Atari environment
    env = gym.make(args.env)
    env.seed(1)

    # Number of actions in the environment
    num_actions = env.action_space.n

    # Create replay memory obect
    max_size = (window_size+1) if no_replay else 1000000
    memory = ReplayMemory(window_length = window_size, max_size = max_size)
    
    # Create model
    model = create_model(window_size, new_size, num_actions, is_linear = is_linear, 
                        model_name='q_network')
                        
    # Create processors
    processor = AtariPreprocessor(new_size)
    processor_combined = HistoryPreprocessor(window_size)

    # Create policy
    policy = LinearDecayGreedyEpsilonPolicy(GreedyEpsilonPolicy(), attr_name='epsilon', 
                                        start_value=1., end_value=.1,num_steps=1000000)
    
    # Create agent
    batch_size = 1 if no_replay else 32
    t_upd_f = 1 if no_replay else 10000
    QN_Agent = DQNAgent(model=model, num_actions=num_actions, policy=policy, memory=memory,
                    processor=processor, processor_combined = processor_combined,
                    batch_size = batch_size, num_burn_in=5000, gamma=.99, 
                    target_update_freq=t_upd_f, train_frames=3, 
                    is_double = is_double, is_dueling = is_dueling)
    
    # Compile agent
    QN_Agent.compile(Adam(lr=.00025))

    # Interval at which to save/test weights    
    weight_interval = 100000


    ### DO YOU WANT TO TRAIN ?
    train = True
    if train:
        # Call functions for logging during training
        logging = [Save_weights('hw2_' + args.env + '_weights_{step}.h5f', interval=weight_interval)]
        logging += [Log_File('hw2_' + args.env + '_log.json', interval=100)]

        # Train agent
        QN_Agent.fit(env, callbacks=logging, num_steps=4000000, interval = 1000, no_replay=no_replay)
        
        # Save final weights
        QN_Agent.save_weights('hw2_results_' + args.env + '_weights.h5f', overwrite=True)
    
    ### DO YOU WANT TO TEST ?
    test = False
    if test:
    	# env = gym.wrappers.Monitor(env, '/home/ramitha/results_videos/results_nono', force=True)
        # Test agent
        start_weight = 100000
        end_weight = 4000000
        for i in range(start_weight, end_weight + weight_interval, weight_interval):
            filename = 'hw2_' + args.env + '_weights_' + str(i) + '.h5f'
            #filename = '/home/ramitha/results_nono/dqn_SpaceInvaders-v0_weights_noddqn_nodunet_' + str(i) + '.h5f'
            QN_Agent.load_weights(filename)
            QN_Agent.evaluate(env, num_episodes=1)


if __name__ == '__main__':
    main()
