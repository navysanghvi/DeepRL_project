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

from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
import keras.backend as K

from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy
from deeprl_hw2.core import ReplayMemory
from deeprl_hw2.preprocessors import ActionHistoryProcessor,VisualProcessor
from deeprl_hw2.callbacks import Log_File, Save_weights


def create_model(input_length, num_actions):  # noqa: D103
    model = Sequential()
    model.add(Dense(1024, input_shape = input_length))
    model.add(Activation('tanh'))
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(num_actions))
    model.add((Activation('relu')))

def main():  # noqa: D103
    
    num_actions = 9
    history_length = 10
    max_size = 200000

    memory = ReplayMemory(max_size = max_size)
    
    # Create model
    model = create_model()
           
    # VGG16 model
    base_model = VGG16(weights='imagenet')
    VGG_model = Model(input = base_model.input, output = base_model.get_layer('fc1').output)

    # Create processors
    visual_processor = VisualProcessor(VGG_model)
    action_processor = ActionHistoryProcessor(num_actions, history_length = history_length)

    # Create policy
    policy = LinearDecayGreedyEpsilonPolicy(GreedyEpsilonPolicy(), attr_name='epsilon', 
                                        start_value=1., end_value=.1,num_steps=5)
    
    # Create agent
    QN_Agent = DQNAgent(model=model, memory=memory, num_actions=num_actions, 
                        visual_processor=visual_processor, action_processor = action_processor,
                        img_dir = './train2014', policy = policy, gamma=.99, 
                        target_update_freq=10000, num_burn_in=5000, batch_size = 32, 
                        is_double = True, is_dueling = True)
    
    # Compile agent
    QN_Agent.compile(Adam(lr=.00025))

    # To save/test weights, and log data    
    weight_interval = 1000
    weight_dir = 'project_weights_DQN_{episode}.h5f'
    log_interval = 100
    log_dir = 'project_log_DQN.json'


    ### DO YOU WANT TO TRAIN ?
    train = True
    if train:
        # Call functions for logging during training
        logging = [Save_weights('project_weights_{episode}.h5f', interval=weight_interval)]
        logging += [Log_File('project_log.json', interval=100)]

        # Train agent
        QN_Agent.fit(log_dir=log_dir, weight_dir=weight_dir, 
            log_interval=log_interval, weight_interval=weight_interval,
            epochs=10, max_episode_length=100)
        
        # Save final weights
        QN_Agent.save_weights('project_weights_DQN.h5f', overwrite=True)
    
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
