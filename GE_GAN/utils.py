#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 15:21
# @Author  : Chenchen Wei
# @Description:
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import os
import time
import keras
import numpy as np

def gpu_set(gpu_id, memory):
    """Using specify gpu with specific memory to train the model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)


def weights_get(names, input_dim, output_dim, use_regular=False):
    """
    If regularization is used, the model loss function should be changed
    """
    if use_regular:
        w = tf.get_variable(names,
                            [input_dim, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4))
    else:
        w = tf.get_variable(names,
                            [input_dim, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
    return w


def bias_get(names, dim):
    """Get the bias of the model"""
    b = tf.get_variable(names,
                        [dim],
                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    return b


def count_time(func):
    """
    Statistical function runtime decorator
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        print('Function：<' + str(func.__name__) + '>TimeCost：{:.2f} Minute'.format((end_time - start_time) / 60))
        return ret
    return wrapper


def input_sequnece(data, n_in=1, dropnan=True):
    """Serialization"""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i - 1))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def data_process_pre(data_path, ralevant_road_segment, target_segment, sliding_window, split_rate):
    """Load the data, x is the neg_road_segments data, y is the target_segment data"""
    df = pd.read_csv(data_path, index_col=None).values
    neg_vals, tar_vals = df[:, ralevant_road_segment], df[:, target_segment]

    scaler1 = MinMaxScaler(feature_range=(0, 1)) # Standardization
    neg_vals = scaler1.fit_transform(neg_vals)
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    tar_vals = scaler2.fit_transform(tar_vals)

    data_x = input_sequnece(neg_vals, n_in=sliding_window).values
    data_y = input_sequnece(tar_vals, n_in=sliding_window).values
    length = int(data_x.shape[0] * split_rate)
    train_x, test_x = data_x[:length, :], data_x[length:, :]
    train_y, test_y = data_y[:length, :], data_y[length:, :]
    return scaler2, train_x, train_y, test_x, test_y

def sample_Z(m, n):
    """Generate the Random noise with shape [m, n]"""
    return np.random.uniform(-1, 1., size=[m, n])

