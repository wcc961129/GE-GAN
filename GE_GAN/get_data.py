#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 15:25
# @Author  : Chenchen Wei
# @Description:
import gensim

from GE_GAN.deepwalk import *
from GE_GAN.utils import *


def get_one_segment_similar(ge_model, road_segment, select_nums):
    """find the most_similar segment of one segment"""
    similar_segments = ge_model.most_similar(str(road_segment))
    if select_nums > len(similar_segments):
        raise ValueError('Select_nums is too large')
    similar_list = similar_segments[:select_nums]
    res = [int(x[0]) for x in similar_list]
    return res


def get_most_similar_road_segments(adj_path, target_segment,
                                   select_nums,
                                   number_walks=10,
                                   walk_length=40,
                                   window_size=5,
                                   representation_size=64, ):
    if not os.path.exists(adj_path[:-8] + '.bin'):
        print('Not Graph Embedding file')
        ge_model = Graph_embedding(name=adj_path[:-8],
                                   number_walks=number_walks,
                                   walk_length=walk_length,
                                   window_size=window_size,
                                   representation_size=representation_size, )(adj_path)
        # get the grapg embedding features
    else:
        print('Loading graph embedding features file')
        ge_model = gensim.models.KeyedVectors.load_word2vec_format(adj_path[:-8] + '.bin')

    if len(target_segment) == 1:
        return get_one_segment_similar(ge_model, target_segment[0], select_nums)
    else:
        raod0_similar = get_one_segment_similar(ge_model, target_segment[0], select_nums)
        raod1_similar = get_one_segment_similar(ge_model, target_segment[1], select_nums)
        all_similar = list(set(raod0_similar) & set(raod1_similar))
        # The last collection of the relevant adjacency detectors
        if len(all_similar) < 1:
            raise ValueError('No all relevant detectors, should increase the window_size')
        return all_similar


class load_data(object):
    def __init__(self,
                 edge_path,
                 data_path,
                 select_nums,
                 sliding_window=12,
                 split_rate=0.8,
                 number_walks=10,
                 walk_length=40,
                 window_size=5,
                 representation_size=64, ):
        self.edge_path = edge_path
        self.sliding_window = sliding_window
        self.split_rate = split_rate
        self.select_nums = select_nums
        self.data_path = data_path
        self.number_walks = number_walks
        self.walk_length = walk_length
        self.window_size = window_size
        self.representation_size = representation_size

    def _call(self, target_segment):
        neg_segments = get_most_similar_road_segments(self.edge_path, target_segment,
                                                      self.select_nums,
                                                      number_walks=self.number_walks,
                                                      walk_length=self.walk_length,
                                                      window_size=self.window_size,
                                                      representation_size=self.representation_size, )
        scaler, train_x, train_y, test_x, test_y = data_process_pre(self.data_path,
                                                                    neg_segments,
                                                                    target_segment,
                                                                    sliding_window=self.sliding_window,
                                                                    split_rate=self.split_rate)
        print('target_segment:{} most_similar_segments:{}'.format(target_segment, neg_segments))
        print('Train_x shape:{} Train_y shape:{}'.format(train_x.shape, train_y.shape))
        print('Test_x shape:{} Test_y shape:{}'.format(test_x.shape, test_y.shape))
        return scaler, train_x, train_y, test_x, test_y

    def __call__(self, target_segment):
        return self._call(target_segment)
