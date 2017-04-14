import sys
sys.path.append('keras-rl-copy')

import keras.optimizers as optimizers
import keras.backend as K
from keras.layers import Lambda, Input, Layer, Dense
from policy import LinearDecayGreedyEpsilonPolicy, GreedyEpsilonPolicy
from copy import deepcopy
import numpy as np
from keras.models import model_from_config, Sequential, Model
import keras.optimizers as optimizers
from keras.preprocessing import image
import random
from objectives import masked_error
from callbacks import Save_weights, Log_File
from loadData import DataSet

def mean_q(y_true, y_pred): 
    return K.mean(K.max(y_pred, axis=-1))

class DQNAgent(object):

    def __init__(self, model, memory, num_actions, visual_processor = None, action_processor = None,
                 text_processor = None, image_dir = './images/', policy=None, gamma=.99, action_fraction = 0.2, 
                 terminal_reward = 3.0, target_update_freq=10000, num_burn_in=500, batch_size=32, 
                 test_policy=GreedyEpsilonPolicy(0.05), is_double=False, is_dueling=False, *args,**kwargs):
        self.memory = memory                                        # Replay Memory
        self.num_actions = num_actions                              # Number of actions
        self.visual_processor = visual_processor                    # Visual Processor
        self.action_processor = action_processor                    # Action Processor
        self.text_processor = text_processor                        # Text Processor
        self.image_dir = image_dir                                  # Folder of images
        self.policy = policy                                        # Policy during training
        self.gamma = gamma                                          # Discount factor
        self.alpha = action_fraction                                # Fraction for movement of bbox
        self.terminal_reward = terminal_reward                      # Terminal action reward
        self.target_update_freq = int(target_update_freq)           # Frequency of updating target model to training model
        self.num_burn_in = num_burn_in                              # Min samples required in replay buffer
        self.batch_size = batch_size                                # Size of sample from replay buffer
        self.test_policy = test_policy                              # Policy during testing
        self.DDQN = is_double                                       # Whether double DQN is being employed
        self.DuelDQN = is_dueling                                   # Whether dueling DQN is being employed
        self.training = False                                       # Whether we are in training mode
        self.step = 0                                               # Number of steps taken
        self.epoch = 0                                              # Number of epochs in training done
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

  
    ######################################## UPDATE NETWORK ########################################
    def update_policy(self, reward):

        metrics = [np.nan for _ in self.metrics_names]

        # DURING TRAINING: Update only if at least num_burn_in samples are in memory
        if self.training and self.step > self.num_burn_in:
            
            # Get samples from replay memory
            samples = self.memory.sample(self.batch_size)

            # Store (s,a,r,s',is_terminal) information of samples in corresponding arrays
            s_0 = []; s_1 = []; act = []; rew = []; t_1 = []
            for e in samples:
                s_0.append(e.state); s_1.append(e.next_state); act.append(e.action)
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
    def reset_start(self, img):
        bbox = deepcopy((0,0, img.size[0], img.size[1]))
        visual_observation = self.visual_processor.process_bbox(bbox, img)
        return bbox, visual_observation


    ######################################## SELECT ACTION ########################################
    def select_action(self, combined_observation):
        
        # Get corresponding Q values for the combined state
        q_values = (self.model.predict_on_batch(
            np.array([combined_observation]))).flatten()
        
        # Select action according to train or test policy
        p = self.policy if self.training else self.test_policy
        action = p.select_action(q_values=q_values)
        
        # Set recent state and action
        self.recent_observation = combined_observation
        self.recent_action = action

        # Return selected action
        return action


    ####################################### TAKE BBOX STEP #######################################
    def visual_step(self, action, bbox, img_size):
        width_change = self.alpha*(bbox[2] - bbox[0])
        height_change = self.alpha*(bbox[3] - bbox[1])
        
        x1 = bbox[0] - width_change; x2 = bbox[2] + width_change
        y1 = bbox[1] - height_change; y2 = bbox[3] + height_change
        x1 = 0 if x1 < 0 else x1; x2 = img_size[0] if x2 > img_size[0] else x2
        y1 = 0 if y1 < 0 else y1; y2 = img_size[1] if y2 > img_size[1] else y2
        
        x1_ = bbox[0] + width_change; x2_ = bbox[2] - width_change
        y1_ = bbox[1] + height_change; y2_ = bbox[3] - height_change
        
        if(action == 0): # Right
            bbox_next = (x1_, bbox[1], x2, bbox[3])

        if(action == 1): # Left
            bbox_next = (x1, bbox[1], x2_, bbox[3])
        
        if(action == 2): # Down
            bbox_next = (bbox[0], y1_, bbox[2], y2)
        
        if(action == 3): # Up   
            bbox_next = (bbox[0], y1, bbox[2], y2_)

        if(action == 4): # Scale up
            bbox_next = (x1, y1, x2, y2)

        if(action == 5): # Scale down
            bbox_next = (x1_, y1_, x2_, y2_)

        if(action == 6): # Fatter
            bbox_next = (x1, bbox[1], x2, bbox[3])

        if(action == 7): # Taller
            bbox_next = (bbox[0], y1, bbox[2], y2)

        if(action == 8): # Stop
            bbox_next = bbox

        return bbox_next


    ######################################## TRAINING ########################################
    def fit(self, log_dir = './', weight_dir = './', log_interval = 1000, 
            weight_interval = 1000, epochs=10, max_episode_length=100):
        
        # Initializing for training
        self.training = True; self.step = 0; episode = 0
        image_data = DataSet('./data/data_train.json')
        image_data.load(0.8)

        # Logging
        log = Log_File(log_dir, log_interval)
        weights = Save_weights(weight_dir, weight_interval)
        log.metrics_names = self.model.metrics_names

        # logging = callbacks[:]; logging += [Print_progress(interval)]; 
        # logging = Loop_Callbacks(logging)
        # logging.set_model(self); logging._set_env(env)
        # logging.set_params({'num_steps': num_steps,})
        # logging.on_train_begin()

        #### Iterate over number of epochs requested
        for epoch in range(epochs):
            image_data.permute()

            #### Iterate over all sentences in training data
            for image_name, sentence, bbox_gt, text in image_data.list_data:
                episode += 1; log.metrics[episode] = []
                episode_step = 0; episode_reward = 0.

                # Get ground truth in (x1,y1,x2,y2) format
                bbox_gt = (bbox_gt[0], bbox_gt[1], 
                    bbox_gt[0]+bbox_gt[2], bbox_gt[1]+bbox_gt[3])

                # Reset action history at start of new episode
                self.action_processor.reset()
                action_observation = self.action_processor.action_vector

                # Load image
                image_path = self.image_dir + image_name
                img = image.load_img(image_path)

                # Reset bbox and bbox features at start of new episode
                bbox, visual_observation = self.reset_start(img)
                
                # IoU of whole image with ground truth
                self.visual_processor.getIoU(bbox, bbox_gt)

                # Reset whole image features at start of new episode
                image_observation = deepcopy(visual_observation)
                
                # Process sentence for text features at start of new episode
                text_observation = np.array(text).ravel()
                
                # Create combined state observed
                combined_observation = np.concatenate((image_observation, visual_observation, 
                                        action_observation, text_observation))
                
                #### Iterate till terminal action or max steps
                while 1:

                    # Select action, process it for action history part of state
                    action = self.select_action(combined_observation)
                    self.action_processor.process_action(action)
                    action_observation = self.action_processor.action_vector

                    # Execute action, process bounding box for bbox features part of state
                    bbox = self.visual_step(action, bbox, img.size); bbox = deepcopy(bbox)
                    visual_observation = self.visual_processor.process_bbox(bbox, img)
                    
                    # Create combined state observed
                    combined_observation = np.concatenate((image_observation, visual_observation, 
                                            action_observation, text_observation))

                    # Calculate reward
                    terminal_action = 1 if action == self.num_actions else 0
                    IoU_prev = self.visual_processor.IoU
                    reward = self.visual_processor.process_reward(terminal_action,
                        self.terminal_reward, IoU_prev, bbox, bbox_gt)
                    
                    # Append to memory
                    self.memory.append(self.recent_observation, self.recent_action, 
                                        reward, terminal_action)

                    # Update policy
                    metrics = self.update_policy(reward)
                    
                    step_logs = {'metrics': metrics,'episode': episode,}
                    log.on_step_end(episode_step, step_logs)

                    # Update trackers
                    episode_reward += reward
                    episode_step += 1
                    self.step += 1

                    # End of epsiode
                    if action == self.num_actions or episode_step > max_episode_length:
                        self.memory.append(combined_observation, self.num_actions-1, 0.0, 1)
                        episode_logs = {'episode_reward': episode_reward,}
                        log.on_episode_end(episode, episode_logs)
                        weights.on_episode_end(episode, episode_logs)
                        print('End of episode ' + str(episode) + '. Reward = ' + str(episode_reward))
                        episode += 1
                        break

            self.epoch += 1

        # End of training
        log.save_data()


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
