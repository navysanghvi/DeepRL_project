"""Loss functions."""

import tensorflow as tf
import keras.backend as K

def huber_loss(y_true, y_pred, max_grad=1.):
    y_true = tf.to_float(y_true)     # Target Value
    y_pred = tf.to_float(y_pred)     # Predicted Value
    max_grad= tf.to_float(max_grad)  # Maximum possible gradient magnitude
    condition = tf.less(tf.abs(y_true - y_pred), max_grad)
    part_1 = 0.5 * tf.square(y_true - y_pred)
    part_2 = (max_grad * (tf.abs(y_true - y_pred))) - (0.5*tf.square(max_grad))
    return tf.where(condition, part_1, part_2)
    

# Masking loss - will enable us to look only at loss 
# corresponding to actions seen while sampling from 
# replay memory during training
def masked_error(args):
    y_true, y_pred, mask = args
    loss = huber_loss(y_true, y_pred)
    loss *= mask
    return K.sum(loss, axis=-1)
