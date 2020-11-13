#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 10:06
# @Author  : Chenchen Wei
# @Description:
import numpy as np

def points(val, point=2):
    """Keep specific decimals"""
    return np.round(val, point)

def evalute(pre, true):
    """Calculating  error"""
    pre, true = np.asarray(pre), np.asarray(true)
    mae = np.mean(np.abs(pre - true))
    rmse = np.sqrt(np.mean((pre - true) ** 2))
    mape = np.mean(np.abs((pre - true) / true)) * 100
    return points(mae), points(rmse), points(mape)

