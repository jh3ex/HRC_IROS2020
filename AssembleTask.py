# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:18:47 2020

@author: Manufacturing
"""

import numpy as np
import pandas as pd
import copy


class AssembleTask():
    def __init__(self, fpath):
        # Read the assembly infomation from given path
        self.tree_info = pd.read_excel(fpath, index_col=None)
        self.h = self.tree_info['row'].max()  # Height
        self.w = self.tree_info['column_to'].max()  # Width
        self.d = 3  # Depth, fixed for current project
        self.initialization()

    def initialization(self):
        # Construct the assembly tree
        # -1 represents empty
        self.tree = -np.ones((self.h, self.w, self.d))
        
        for _, task in self.tree_info.iterrows():
            # First layer: finish time
            self.tree[task['row']-1, task['column_from']-1, 0] = task['finish_time']
            
            # Second layer: robot or human
            # 2: robot, 3: human or robot, 4: human
            # 1: reserved for human busy
            # 0: reserved for robot busy
            self.tree[task['row']-1, task['column_from']-1, 1] = task['eligibility'] + 3
            
            # Third layer: columns for a task
            self.tree[task['row']-1, task['column_from']-1, 2] = task['column_to'] - task['column_from'] + 1
            
#            # Fourth layer: original task row number
#            self.tree[task['row']-1, :, 3] = task['row'] - 1
        return self.tree
    
    def next_state(self, state, action, robot):
        # Generate subsequent state given current state and action
        
#        simul_task_time = -np.infty
        other_free = False
        
        if robot:
            # If it is a robot action
            # Change task to robot ongoing status
            state[0, action, 1] = 0
            # Test if there is human ongoing status
            simul_task = (state[0, :, 1] == 1).any()
            if simul_task:
                # If there is human ongoing task, get the task status
                simul_action = np.where(state[0, :, 1] == 1)[0][0]
                simul_task_time = state[0, simul_action, 0]
            elif not simul_task:
                # Else, human is available
                if (state[0, :, 1] == 3).any() or (state[0, :, 1] == 4).any():
                    other_free = True  # We have free agent and available task
        
        elif not robot:
            # If it is a human action
            # Change task to human ongoing status
            state[0, action, 1] = 1
            # Test if there is human ongoing status
            simul_task = (state[0, :, 1] == 0).any()
            if simul_task:
                # If there is robot ongoing task, get the task status
                simul_action = np.where(state[0, :, 1] == 0)[0][0]
                simul_task_time = state[0, simul_action, 0]
            elif not simul_task:
                # Else, robot is available
                if (state[0, :, 1] == 2).any() or (state[0, :, 1] == 3).any():
                    other_free = True  # We have free agent and available task
        
        # Finish time of the chosen task
        task_time = state[0, action, 0]
        
        
        if other_free:
            # If there is free agent and available task,
            # we must await for a decision.
            # There is no elapsed time.
            elapsed_time = 0
            terminal = False
#            return state, elapsed_time
        elif not other_free:
            # If the decision on the other agent is not needed
            if simul_task:
                # We need to determine who finishes first                
                if task_time < simul_task_time:
                    # If current agent finish first
                    elapsed_time = task_time
                    state[0, simul_action, 0] -= elapsed_time
                    state, terminal = self.drop(state, action, int(state[0, action, 2]))
                    # Now, current agent finishes its job
                    # See if current agent has anything to do
                    alist = self.available_action(state, robot)
                    if not alist:
                        # If the agent finishes its job
                        # but there is nothing he can do
                        # then this is not an actual 'next state'
                        state, terminal = self.drop(state, simul_action, int(state[0, simul_action, 2]))
                        elapsed_time = simul_task_time
                    
                elif task_time > simul_task_time:
                    # If the other agent finishes first
                    elapsed_time = simul_task_time
                    state[0, action, 0] -= elapsed_time
                    state, terminal = self.drop(state, simul_action, int(state[0, simul_action, 2]))
                    alist = self.available_action(state, not robot)
                    if not alist:
                        # If the other agent finishes its job
                        # but there is nothing he can do
                        # then this is not an actual 'next state'
                        state, terminal = self.drop(state, action, int(state[0, action, 2]))
                        elapsed_time = task_time
                    
                elif task_time == simul_task_time:
                    # If they finish at the same time
                    elapsed_time = simul_task_time
                    state, terminal = self.drop(state, simul_action, int(state[0, simul_action, 2]))
                    state, terminal = self.drop(state, action, int(state[0, action, 2]))
            
            elif not simul_task:
                elapsed_time = task_time
                state, terminal = self.drop(state, action, int(state[0, action, 2]))
        
        
        
        return state, terminal, elapsed_time
            
            
    def drop(self, state, column_from, column_to):
        
        # Eliminate the finished task in last row
        state[0, column_from:column_to+column_from, :] = -2
        
        while True:
            state_old = copy.deepcopy(state)
            # Keep the state before dropping
            for row in range(1, self.h):
                for col in range(self.w):
                    if (state[row, col, 1] != -1) and (state[row, col, 1] != -2):
                        # If the col contains a task
                        if (state[row-1, col:col+int(state[row, col, 2]), 0] == -2).all():
                            # If the previous row is empty
                            # regarding the corresponding position

                            state[row-1, col:col+int(state[row, col, 2]), :] =\
                            state[row, col:col+int(state[row, col, 2]), :]
                            
                            state[row, col:col+int(state[row, col, 2]), :] = -2
                            
            if (state == state_old).all():
                # If there is no change after droping, break
                break
        
        terminal = False # Indicates if the state is terminal
        if (state == -2).all():
            terminal = True
        return state, terminal
    
    def available_action(self, state, robot):
        # Find all the elgibible actions for a given state        
        alist = []  # Contains all elgibible actions
        
        if robot:
            if not (state[0, :, 1] == 0).any():
                for i in range(self.w):
                    if (state[0, i, 1] == 2) or (state[0, i, 1] == 3):
                        alist.append(i)
        
        elif not robot:
            if not (state[0, :, 1] == 1).any():
                for i in range(self.w):
                    if (state[0, i, 1] == 3) or (state[0, i, 1] == 4):
                        alist.append(i)
        


        return alist
    
    


if __name__ == '__main__':
    
    fpath = './task_1.xlsx'
    #
    at1 = AssembleTask(fpath)
    state = at1.tree
    #
    #state[0, 4, 0] = 4
    #state[0, 0, 1] = 1
    #
    alist = at1.available_action(state, True)
    print('Actions: ', alist)
    ##print('before 0\n', state[:,:,1])
    ##state, terminal = at1.drop(at1.tree, 0, 4)
    ##print('after 1\n', state[:,:,1])
    ##state, terminal = at1.drop(state, 4, 8)
    ##print('after 2\n', state[:,:,1])
    
    
    #print('before\n', state[:, :, 1])
    state, terminal, elapsed_time = at1.next_state(state, 4, True)
    print(elapsed_time)
    #print('after\n', state[:, :, 1])
    ##
    ##alist = at1.available_action(state, True)
    ##print(alist)
    ##alist = at1.available_action(state, False)
    ##print(alist)
    #
