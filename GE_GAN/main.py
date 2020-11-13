#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 19:22
# @Author  : Chenchen Wei
# @Description:
# @Contact Email: 771792694@qq.com
import sys

from GE_GAN.model import *
from GE_GAN.result_save import *
from GE_GAN.visualization import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# dataset params
data_params = {'pems': {'data_path': './data/',
                        'file_names': ['combine_E_workday_n0.csv',
                                       'combine_E_weekend_n0.csv'],
                        'edge_path': './data/pems_24_point.adjlist', },
               # "adjlist" file represents the connection between nodes
               'seattle': {'data_path': './data/',
                           'file_names': ['workday_seattle.csv',
                                          'weekend_seattle.csv'],
                           'edge_path': './data/seattle_traffic.adjlist', }, }
# all params
model_params = {'epoch': 20000,
                'sliding_windows': [12],
                'batch_size': 512,
                'ge_units': [512, 256, 128],
                'dis_units': [512, 256, 128],
                'select_nums': 4,  # 4 for pems data,  10 for seattle data
                'number_walks': 10,  # the iterate times of random walk
                'walk_length': 40,  # 40 for pems data, 100 for seattle data
                'window_size': 5,  # selected window size
                'representation_size': 64,  # embedding size
                }
# data params
flags = ['seattle', 'pems']
data_flag = flags[1]
data_path, file_names, edge_path = data_params[data_flag].values()
file_name = file_names[1]
data_path = os.path.join(data_path, file_name)
# graph embedding params
select_nums = model_params['select_nums']
number_walks = model_params['number_walks'],
walk_length = model_params['walk_length'],
window_size = model_params['window_size'],
representation_size = model_params['representation_size']
# gan params
epoch = model_params['epoch']
batch_size = model_params['batch_size']
ge_units = model_params['ge_units']
dis_units = model_params['dis_units']
sliding_windows = model_params['sliding_windows']
model_name = 'GE-GAN'
target_segement = [6]
# If use GPU
gpu_set(0, 0.95)

for sliding_window in sliding_windows:
    for _ in range(1):  # repeat x times
        tf.reset_default_graph()  # reset the tensorflow graph
        save_path = 'Results/{}/{}/slw={} ep={}'.format(model_name,
                                                        file_name[:-4],
                                                        sliding_window,
                                                        epoch)
        if not os.path.exists(save_path):  # create save directory
            os.makedirs(save_path)
        all_data = load_data(edge_path=edge_path,
                             data_path=data_path,
                             select_nums=select_nums,
                             sliding_window=sliding_window,
                             number_walks=number_walks,
                             walk_length=walk_length,
                             window_size=window_size,
                             representation_size=representation_size,
                             )(target_segement)  # return train and test set
        # input data is the neg_segements data, to gene the target_segement data
        with tf.Session() as sess:
            model = GE_GAN(sess=sess,
                           all_data=all_data,
                           target_nums=len(target_segement),
                           ge_hidden_lists=ge_units,
                           dis_hidden_lists=dis_units,
                           epoch=epoch,
                           batch_size=batch_size,
                           save_model=True,
                           save_model_path=save_path)
            gene, true = model.train()
        # The params will be saved
        params = {'target_segement': target_segement,
                  'ep': epoch,
                  'bs': batch_size,
                  'g_hi': ge_units,
                  'd_hi': dis_units,
                  'number_walks': number_walks,
                  'walk_length': walk_length,
                  'window_size': window_size,
                  'representation_size': representation_size,
                  'sliding_window': sliding_window, }
        # Saving the generated and observed data,
        # Also save the results with the params, in 'results.csv'
        save_results(save_path=save_path,
                     model_name=model_name,
                     data_name=file_name[:-4],
                     params=params,
                     gene=gene,
                     true=true).main()
        # visualizatation of one target_segement
        mae, rmse, mape = evalute(gene[:, 0], true[:, 0])
        plt_figures([gene[:, 0], true[:, 0]], ['gene', 'true'],
                    save_path=save_path,
                    title='road={} mae={} rmse={} mape={}%'.format(target_segement[0], mae, rmse, mape),
                    save_names='gene_true.png')
