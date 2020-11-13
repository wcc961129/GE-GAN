#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 16:03
# @Author  : Chenchen Wei
# @Description:
import os
import time
from GE_GAN.evalutation import *
import csv
import pandas as pd


class save_results(object):
    """Save the generated and observed data,
    Save the model and data parameters"""
    def __init__(self, save_path, model_name, data_name, params, gene, true,
                 csv_path = 'result.csv'):
        self.save_path = save_path
        self.model_name = model_name
        self.data_name = data_name
        self.params = params
        self.gene = gene
        self.true =true
        self.csv_path = csv_path

    def main(self):
        mae, rmse, mape = evalute(self.gene, self.true) # Error
        now_times = time.strftime("%m-%d %H:%M:%S", time.localtime()) # Record time
        self.save_csv([self.gene, self.true], ['gene.csv', 'true.csv'])
        rows = [self.model_name, self.data_name, mae, rmse, mape
                ] + list(self.params.values()) + [now_times]
        print(rows)  #Experiment results and model parameters
        self.write_csv(rows) #Save results

    def save_csv(self, vals, names):
        """save muti-csv files"""
        for val, name in zip(vals, names):
            pd.DataFrame(val).to_csv(os.path.join(self.save_path, name),
                                     header=None, index=None)

    def write_csv(self, rows):
        """Save rows to the specified csv file"""
        with open(self.csv_path, 'a+', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(rows)