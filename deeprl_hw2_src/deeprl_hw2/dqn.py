import sys
sys.path.append('keras-rl-copy')

import keras.optimizers as optimizers
import keras.backend as K
from keras.layers import Lambda, Input, Layer, Dense
from policy import LinearDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy
from callbacks import Print_progress,Loop_Callbacks,Print_test_progress
from copy import deepcopy
import numpy as np
from keras.models import model_from_config, Sequential, Model
import keras.optimizers as optimizers
from objectives import masked_error

def mean_q(y_true, y_pred): 
    return K.mean(K.max(y_pred, axis=-1))

class DQNAgent(object):

    def __init__(self,model,memory,num_actions,processor = None,policy=None,gamma=.99,target_update_freq=10000,num_burn_in=1000,
                 batch_size=32,train_frames=1,test_policy=GreedyEpsilonPolicy(0.05), is_double=True, 
                 is_dueling=False,processor_combined = None, *args,**kwargs):
        self.processor = processor                                  # Atari Processor
        self.processor_combined = processor_combined                # History Processor
        self.memory = memory                                        # Replay Memory
        self.num_actions = num_actions                              # Number of actions
        self.policy = policy                                        # Policy during training
        self.gamma = gamma                                          # Discount factor
        self.target_update_freq = int(target_update_freq)           # Frequency of updating target model to training model
        self.num_burn_in = num_burn_in                              # Min samples required in replay buffer
        self.batch_size = batch_size                                # Size of sample from replay buffer
        self.train_frames = train_frames                            # Training model update frequency
        self.test_policy = test_policy                              # Policy during testing
        self.DDQN = is_double                                       # Whether double DQN is being employed
        self.DuelDQN = is_dueling                                   # Whether dueling DQN is being employed
        self.training = False                                       # Whether we are in training mode
        self.step = 0                                               # Number of steps taken
        self.compiled = False                                       # Whether the agent has been compiled

        # Modify the model in case of dueling: 
        # Remove the last layer; create a new layer of V (1 unit) and A (num_action units);
        # Add output layer corresponding to V + A(a) - mean(A) for each action a.
        if self.DuelDQN:
            layer = model.layers[-2]
            num_action = model.output._keras_shape[-1]
            y = Dense(num_action + 1, activation='linear')(layer.output)
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), 
                                 output_shape=(num_actions,))(y)
            model = Model(input=model.input, output=outputlayer)
        self.model = model


    ####################################### COMPILING ########################################    
    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]    
        
        # Compile target model
        config = {'class_name': self.model.__class__.__name__,'config': self.model.get_config(),}
        self.target_model = model_from_config(config)
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.compile(optimizer='sgd', loss='mse')
        
        # Compile model
        self.model.compile(optimizer='sgd', loss='mse')
        
        # We create a 'trainable_model', where:
        # Input is [input of model, desired output of model, mask], 
        # Output is [loss, output of model],
        # Loss is calculated between desired output and output of model, in a lambda layer
        y_true = Input(name='y_true', shape=(self.num_actions,))
        y_pred = self.model.output
        mask = Input(name='mask', shape=(self.num_actions,))
        loss_out = Lambda(masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
        ins = [self.model.input]
        trainable_model = Model(input=ins + [y_true, mask], output=[loss_out, y_pred])

        # Compile trainable_model
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        # Agent has now been compiled
        self.compiled = True


    ######################################## SELECT ACTION ########################################
    def select_action(self, state):

        # Get combined state according to window length specified
        state_combined = self.processor_combined.process_state_for_network(state,
            self.memory.recent_observations,self.memory.recent_terminals)
        
        # Get corresponding Q values for the combined state
        q_values = self.calc_q_values(state_combined)
        
        # Select action according to train or test policy
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)
        
        # Set recent state and action
        self.recent_observation = state
        self.recent_action = action

        # Return selected action
        return action
     
    def calc_q_values(self, state):
        batch = np.array([state])
        batch = self.processor.process_batch(batch)
        qval = self.model.predict_on_batch(batch)
        qval = qval.flatten()
        return qval

  
    ######################################## UPDATE NETWORK ########################################
    def update_policy(self, reward, terminal, no_replay=False,):
        self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
        metrics = [np.nan for _ in self.metrics_names]

        # DURING TRAINING: Update on every 'train_frames'th frame seen, and 
        #                  only if at least num_burn_in samples are in memory
        if self.training and self.step > self.num_burn_in and self.step % self.train_frames == 0:
            
            # Get samples from replay memory (no replay if no_replay is active)
            if(no_replay):
            	samples = self.memory.sample(1, indexes = [self.memory.window_length])
            else :
            	samples = self.memory.sample(self.batch_size)

            # Store (s,a,r,s',is_terminal) information of samples in corresponding arrays
            s_0 = []; s_1 = []; act = []; rew = []; t_1 = []
            for e in samples:
                s_0.append(e.state); s_1.append(e.next_state); act.append(e.action); 
                rew.append(e.reward); t_1.append(0. if e.is_terminal else 1.)
            s_0=np.array(s_0); s_0=self.processor.process_batch(s_0)
            s_1=np.array(s_1); s_1=self.processor.process_batch(s_1)
            rew = np.array(rew); t_1 = np.array(t_1)

            # Get Q values from samples
            if self.DDQN:
                # For double DQN
                Q_1 = self.model.predict_on_batch(s_1)
                Q = self.target_model.predict_on_batch(s_1)
                Q = Q[range(self.batch_size), np.argmax(Q_1, axis=1)]
            else:
                # No double DQN
                Q = self.target_model.predict_on_batch(s_1)
                Q = np.max(Q, axis=1).flatten()

            # Target Q values after discounting for samples
            target_Q = rew + ((self.gamma * Q) * t_1)

            # Build inputs and desired outputs for trainable_model
            target_Q_mat = np.zeros((self.batch_size, self.num_actions))
            target_l = np.zeros((self.batch_size,))
            mask_mat = np.zeros((self.batch_size, self.num_actions))
            for idx, (target, mask, q, a) in enumerate(zip(target_Q_mat, mask_mat, target_Q, act)):
                target[a] = q; target_l[idx] = q; mask[a] = 1. 
            target_Q_mat = np.array(target_Q_mat).astype('float32')
            mask_mat = np.array(mask_mat).astype('float32')
            ins = [s_0] if type(self.model.input) is not list else s_0

            # Update model by training on samples selected
            metrics = self.trainable_model.train_on_batch(ins + [target_Q_mat, mask_mat], [target_l, target_Q_mat])
            #print self.model.metrics_names
            
            # Metrics for logs
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  

        # DURING TRAINING: Do a hard update of target network weights acccording to set frequency
        if self.training and self.step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        return metrics

    @property
    def metrics_names(self):
        # Discard individual losses
        assert len(self.trainable_model.output_names) == 2
        nn_name = self.trainable_model.output_names[1]
        #print nn_name
        model_metrics = [name for i, name in enumerate(self.trainable_model.metrics_names) if i not in (1, 2)]
        model_metrics = [name.replace(nn_name + '_', '') for name in model_metrics]
        return model_metrics
    
    @property
    def linear_eps_track(self):
        return self.__policy
    
    @linear_eps_track.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy.track_agent(self)
    
    
     
    ######################################## RESET START ########################################
    def reset_start(self,env):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()
        observation = deepcopy(env.reset())
        observation = self.processor.process_state_for_memory(observation)
        return observation


    ######################################## TRAINING ########################################
    def fit(self, env, num_steps, callbacks=None, interval=10000, no_replay = False):
        
        # Initializing for training
        self.training = True; self.step = 0
        episode = 0; observation = None
        episode_reward = None; episode_step = None

        # Logging
        logging = callbacks[:]; logging += [Print_progress(interval)]; 
        logging = Loop_Callbacks(logging)
        logging.set_model(self); logging._set_env(env)
        logging.set_params({'num_steps': num_steps,})
        logging.on_train_begin()

        # Iterate over number of steps requested
        while self.step < num_steps:
                if observation is None:  
                    logging.on_episode_begin(episode)
                    episode_step = 0; episode_reward = 0.
                    observation = self.reset_start(env)
                
                logging.on_step_begin(episode_step)

                action = self.select_action(observation)
                done = False		
                observation, reward, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                   observation, reward = self.processor.process_step(observation, reward)
                
                metrics = self.update_policy(reward,terminal=done, no_replay=no_replay)
                step_logs = {'metrics': metrics,'episode': episode,}
                logging.on_step_end(episode_step, step_logs)

                # Update trackers
                episode_reward += reward
                episode_step += 1
                self.step += 1

                # End of epsiode
                if done:
                    self.select_action(observation)
                    self.update_policy(0.,no_replay=no_replay,terminal=done)
                    episode_logs = {'episode_reward': episode_reward,}
                    logging.on_episode_end(episode, episode_logs)
                    episode += 1; observation = None
                    episode_step = None; episode_reward = None

        # End of training
        logging.on_train_end(logs={'ended': True})


    ######################################## TESTING ########################################
    def evaluate(self, env, num_episodes = 1):
        
        # Initialize for testing
        self.training = False; self.step = 0
        
        # Logging
        logging = [Print_test_progress()]; logging = Loop_Callbacks(logging)
        logging.set_model(self); logging._set_env(env)
        logging.set_params({'num_episodes': num_episodes,})
        logging.on_train_begin()
        
        # Track total rewards
        total_reward = 0.; reward_list = []
        
        # Iterate over number of episodes requested
        for episode in range(num_episodes):
            episode_reward = 0.; episode_step = 0
            observation = self.reset_start(env)
            done = False

            # Iterate till all lives lost
            while not done:
                action = self.select_action(observation)
                reward = 0.
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                   observation, r = self.processor.process_step(observation, r)
                reward += r
                self.update_policy(reward, terminal=done)
                episode_reward += reward
                episode_step += 1
                self.step += 1
            self.select_action(observation)
            self.update_policy(0., terminal=done)
            episode_logs = {'episode_reward': episode_reward,'num_steps': episode_step,}
            logging.on_episode_end(episode, episode_logs)

            # Append episode results
            total_reward += episode_reward
            reward_list.append(episode_reward)
            
        # Print final results
        print('Mean reward = ' + str(total_reward/(episode+1)))
        print('Standard Deviation = ' + str(np.std(reward_list)))
        logging.on_train_end()
    

    ######################################## WEIGHTS ########################################
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)
