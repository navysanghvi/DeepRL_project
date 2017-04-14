"""Suggested Preprocessors."""

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

class ActionHistoryProcessor:
    def __init__(self, num_actions, history_length=1):
        self.num_actions = num_actions
        self.history_length = history_length
        self.action_number = 0
        self.action_history = np.zeros(self.action_number).astype('int')
        self.action_vector = self.get_action_vector()

    def process_action(self, latest_action):
        if self.action_number < self.history_length:
            self.action_number += 1
            self.action_history = np.insert(self.action_history, 0, latest_action)
        else:
            self.action_history = np.insert(
                np.delete(self.action_history, self.action_number-1), 0, latest_action)
        self.action_vector = self.get_action_vector()

    def get_action_vector(self):
        action_matrix = np.zeros((self.history_length, self.num_actions))
        action_matrix[range(self.history_length-self.action_number,self.history_length), 
                          self.action_history] = 1
        action_vector = action_matrix.ravel()
        return action_vector

    def reset(self):
        self.action_number = 0
        self.action_history = np.zeros(self.action_number).astype('int')
        self.action_vector = self.get_action_vector()


        
class VisualProcessor:
    def __init__(self, VGG_model, IoU_thresh = 0.6, IoU_tol = 1e-3, vgg_size = (224,224)):
        self.vgg_size = vgg_size
        self.VGG_model = VGG_model
        self.IoU_thresh = IoU_thresh
        self.IoU_tol = IoU_tol
        self.IoU = 0.0

    def process_bbox(self, bbox, img):
        img = (img.crop(bbox)).resize(self.vgg_size)
        fc1_features = self.VGG_model.predict(preprocess_input(
            np.expand_dims(image.img_to_array(img), axis=0)))
        return fc1_features

    def process_reward(self, terminal_action, terminal_reward, IoU_prev, bbox, bbox_gt):
        self.getIoU(bbox, bbox_gt)
        if terminal_action:
            reward = terminal_reward if self.IoU > self.IoU_thresh else -terminal_reward
        elif abs(self.IoU - IoU_prev) < self.IoU_tol:
            reward = 0
        else:
            reward = 1 if self.IoU > IoU_prev else -1
        return reward
    
    def getIoU(self, bbox, bbox_gt):
        w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
        w_gt = bbox_gt[2]-bbox_gt[0]; h_gt = bbox_gt[3]-bbox_gt[1]
        inter_x1 = max(bbox[0], bbox_gt[0])
        inter_y1 = max(bbox[1], bbox_gt[1])
        inter_x2 = min(bbox[2]-1, bbox_gt[2]-1)
        inter_y2 = min(bbox[3]-1, bbox_gt[3]-1)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = w*h + w_gt*h_gt - inter
        self.IoU = float(inter)/union


class TextProcessor:
    def __init__(self, text_model):
        self.text_model = text_model

    def process_sentence(self, sentence):
        #TODO
        return text_features
