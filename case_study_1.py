# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:59:05 2020

@author: Manufacturing
"""

import numpy as np
from AssembleTask import AssembleTask
from NN import NeuralNet
from MTCS import MonteCarloTreeSearch
from VIS import Visualization


#class MonteCarloTreeSearch():
#    def __init__(self, MaxDepth, MaxIters, MaxSearches, c_puct, atask, robot_net, human_net, switch, tau)

fpath = './task_1.xlsx'

at1 = AssembleTask(fpath)

#np.random.seed(seed=999)
#
#robot_net = NeuralNet(at1.h, at1.w, at1.d, 16, 1, value_norm=110)
#human_net = NeuralNet(at1.h, at1.w, at1.d, 16, 1, value_norm=90)
#
#mtcs1 = MonteCarloTreeSearch(MaxDepth=3,
#                             MaxSearches=30,
#                             MaxIters=10,
#                             c_puct=100,
#                             atask=at1,
#                             robot_net=robot_net,
#                             human_net=human_net,
#                             switch=5,
#                             tau=1000,
#                             data_size=200
#                             )
#
#
vis = Visualization()
###
vis.show_image(at1.tree, 'Initialization')
#
##state, terminal = mtcs1.robot_search(at1.tree)
#
##
#mtcs1.mixed_iterations()
##mtcs1.robot_iterations()
#
##mtcs1.human_iterations()
#completion_time, computing_time = mtcs1.simulation(create_video=False)
#
#completion_time, computing_time = mtcs1.mixed_simulation(create_video=True)



