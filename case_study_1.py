# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:59:05 2020

Main program for implementing all the codes

@author: Manufacturing
"""

import numpy as np
from AssembleTask import AssembleTask
from NN import NeuralNet
from MTCS import MonteCarloTreeSearch

# Given the path of the assembly task
fpath = './task_1.xlsx'

# Creat the assembly chessboard via AssembleTask
at1 = AssembleTask(fpath)

# Set random seed
np.random.seed(seed=999)

# Create neural network for robot and human
# Note: the architecure of the neural network
#       is given inside the class, one should
#       go to NN.py if changes are needed
robot_net = NeuralNet(at1.h, at1.w, at1.d, 16, 1, value_norm=110)
human_net = NeuralNet(at1.h, at1.w, at1.d, 16, 1, value_norm=90)

# Create the MTCS object
mtcs1 = MonteCarloTreeSearch(MaxDepth=3,
                             MaxSearches=30,
                             MaxIters=10,
                             c_puct=100,
                             atask=at1,
                             robot_net=robot_net,
                             human_net=human_net,
                             switch=5,
                             tau=1000,
                             data_size=200
                             )

# Train the human and robot agent
# in a mixed way, i.e. train robot
# agent for 'switch' times and then
# train human agent and continues
mtcs1.mixed_iterations()

# After the training is completed,
# do simulation to see the result
completion_time, computing_time = mtcs1.mixed_simulation(create_video=True)


## You can also choose to train one
## of the agents only
##mtcs1.robot_iterations()
#
##mtcs1.human_iterations()
#completion_time, computing_time = mtcs1.simulation(create_video=False)
#

