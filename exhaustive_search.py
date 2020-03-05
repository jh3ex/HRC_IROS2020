# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:00:07 2020

Exhaustive search, for comparison purpose

@author: Manufacturing
"""

import numpy as np
import copy
import time

class TunaSearch:
    def __init__(self, atask):
        # Assmebly task
        self.atask = atask
        # Get the dimension of the task
        self.h, self.w, self.d =self.atask.tree.shape
        
        # List of completion time for all routes
        self.completion_time = []
        
        
    def iterations_0(self):
        # Search starts from the initial state
        state = copy.deepcopy(self.atask.tree)
        
        # Initial node pool
        #   node_pool[i][0]: state tensor of ith node
        #   node_pool[i][1]: elapsed time to current state
        #   node_pool[i][2]: True/False, if terminal
        node_pool = [[state, 0, False]]
        
        
        
        while True:
            # Begin search
            start_time = time.time()
            node_pool, terminal = self.search(node_pool)
            
            end_time = time.time()
            
            
            print(len(node_pool), ', Elapsed Time: ', end_time - start_time)
            
            f1=open('tunasearch_record.txt', 'a')
            f1.write(str(len(node_pool)) + ', ' + str(end_time - start_time) + '\n')
            f1.close()
            
            if terminal:
                break
        
        
        
        
        

    def iterations(self):
        # Search starts from the initial state
        state = copy.deepcopy(self.atask.tree)
        
        # Initial node pool
        #   node_pool[i][0]: state tensor of ith node
        #   node_pool[i][1]: elapsed time to current state
        #   node_pool[i][2]: True/False, if terminal
        node_pool = [[state, 0, False]]
        
        start_time = time.time()
        
        while True:
            # Begin search
            node_pool, terminal = self.search(node_pool)
            
            
            print(len(node_pool))
            
            if len(node_pool) > 1000:
                
                idx = list(np.random.choice(range(len(node_pool)), size=1000, replace=False))
                
                node_pool = [node_pool[x] for x in idx]
            
            if terminal:
                break
        
        
        end_time = time.time()
        print('End || Elapsed Time: ', end_time - start_time)
        
        
    
    def search(self, node_pool):
        
        # node = [state, time_elapsed, terminal]
        # Create a new node pool
        new_node_pool = []
        # If all the states are terminal states
        all_terminal = True
        
        for node in node_pool:
            
            state = copy.deepcopy(node[0])
            
            
            alist_human = self.atask.available_action(state, robot=False)
            
            if alist_human:
                for action in alist_human:
                    state = copy.deepcopy(node[0])
                    
                    new_state, terminal, elapsed_time =\
                    self.atask.next_state(state, action, robot=False)
                    comp_time = elapsed_time + node[1]
                    
                    
                    if terminal:
                        self.completion_time.append(comp_time)
                    
                    else:
                        new_node_pool.append([new_state, comp_time, terminal])
                        all_terminal = False
                    
            elif not alist_human:
                alist_robot = self.atask.available_action(state, robot=True)
                for action in alist_robot:
                    state = copy.deepcopy(node[0])
                    
                    new_state, terminal, elapsed_time =\
                    self.atask.next_state(state, action, robot=True)
                    comp_time = elapsed_time + node[1]
                    
                    
                    
                    if terminal:
                        self.completion_time.append(comp_time)
                    else:
                        new_node_pool.append([new_state, comp_time, terminal])
                        all_terminal = False
                            
                            

               
        return new_node_pool, all_terminal
    
    
    def iterations_2(self):
        # Search starts from the initial state
        state = copy.deepcopy(self.atask.tree)
        
        # Initial node pool
        #   node_pool[i][0]: state tensor of ith node
        #   node_pool[i][1]: elapsed time to current state
        #   node_pool[i][2]: True/False, if terminal
        node_pool = [[state, [0], False]]
        
        start_time = time.time()
        
        while True:
            # Begin search
#            node_pool, terminal = self.search(node_pool)
            node_pool, terminal = self.search_2(node_pool)
            
            for node in node_pool:
                node[1] = self.__list(node[1])
                
                
            print(len(node_pool))
            
            
            if terminal:
                break
        
        
        end_time = time.time()
        print('End || Elapsed Time: ', end_time - start_time)        
            
    def search_2(self, node_pool):
        
        # node = [state, time_elapsed, terminal]
        # Create a new node pool
        new_node_pool = []
        # If all the states are terminal states
        all_terminal = True
        
        for node in node_pool:
            
            state = copy.deepcopy(node[0])
            
            
            alist_human = self.atask.available_action(state, robot=False)
            
            if alist_human:
                for action in alist_human:
                    state = copy.deepcopy(node[0])
                    
                    new_state, terminal, elapsed_time =\
                    self.atask.next_state(state, action, robot=False)
                    
                    
                    
                    if terminal:
                        self.completion_time.append([x+elapsed_time for x in node[1]])
                    else:
                        all_terminal = False
                        visited = False
                        for nn in new_node_pool:
                            if (nn[0] == new_state).all():
                                visited = True
                                nn[1] += [x+elapsed_time for x in node[1]]
                                break
                            
                        if not visited:
                                
                            new_node_pool.append([new_state,
                                                  [x+elapsed_time for x in node[1]],
                                                  terminal])
                            
                    
            elif not alist_human:
                alist_robot = self.atask.available_action(state, robot=True)
                for action in alist_robot:
                    state = copy.deepcopy(node[0])
                    
                    new_state, terminal, elapsed_time =\
                    self.atask.next_state(state, action, robot=True)
                    
                    
                    
                    
                    if terminal:
                        self.completion_time.append([x+elapsed_time for x in node[1]])
                    else:
                        all_terminal = False
                        visited = False
                        for nn in new_node_pool:
                            if (nn[0] == new_state).all():
                                visited = True
                                nn[1] += [x+elapsed_time for x in node[1]]
                                break
                            
                        if not visited:
                                
                            new_node_pool.append([new_state,
                                                  [x+elapsed_time for x in node[1]],
                                                  terminal])
                            

               
        return new_node_pool, all_terminal       
        
        
    def __list(self, x):
        return list(dict.fromkeys(x))
        
    

if __name__ == '__main__':
    from AssembleTask import AssembleTask
    
    fpath = './task_1.xlsx'
    at1 = AssembleTask(fpath)
    ts1 = TunaSearch(at1)
    
    ts1.iterations_0()
    







