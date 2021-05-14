import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from custom_layers.bjorck_dense import BjorckDense
from custom_activations.activations import group_sort


def get_bjorck_discriminator(input_shape=1, depth=5, width=64, bjorck_beta=0.5, bjorck_iter=5, bjorck_order=2,
                             group_size=2):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    assert depth > 0
    for _ in range(depth):
        model.add(BjorckDense(width, activation=lambda x: group_sort(x, group_size=group_size),
                              bjorck_beta=bjorck_beta, bjorck_iter=bjorck_iter,
                              bjorck_order=bjorck_order
                              ))
    model.add(BjorckDense(1, activation=None))

    return model


def get_clipped_discriminator(input_shape=1, depth=5, width=64):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    for _ in range(depth):
        model.add(Dense(width, activation='relu',
                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01),
                        kernel_constraint=MinMaxValue(minval=-0.01, maxval=0.01)))
    model.add(Dense(1, activation=None))

    return model


class MinMaxValue(tf.keras.constraints.Constraint):
    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, w):
        return tf.clip_by_value(w, clip_value_min=self.minval, clip_value_max=self.maxval)

    def get_config(self):
        return {'minval': self.minval, 'maxval': self.maxval}
