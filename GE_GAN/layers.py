#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 15:52
# @Author  : Chenchen Wei
# @Description:
from GE_GAN.utils import *


class Linear:
    """return y = x*w+b"""
    def __init__(self, name, input_dim, output_dim, use_bias=True, dropout=0., act=tf.nn.relu):
        self.dropout = dropout
        self.act = act
        self.w = weights_get(name + '_w', input_dim, output_dim)
        self.use_bias = use_bias
        if self.use_bias:
            self.b = bias_get(name + '_b', output_dim)

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, rate=self.dropout)
        if self.use_bias:
            x = tf.matmul(x, self.w) + self.b
        else:
            x = tf.matmul(x, self.w)
        x = self.act(x)
        return x

    def __call__(self, inputs):
        return self._call(inputs)
