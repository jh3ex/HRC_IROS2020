# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:08:54 2020

@author: jingh
"""
import numpy as np

class RingBuffer:
    def __init__(self, size):
        self.state = [None] * (size + 1)
        self.action = [None] * (size + 1)
        self.value =  [None] * (size + 1)
        self.policy =  [None] * (size + 1)
        
        
        self.start = 0
        self.end = 0


    def append(self, state, action, value, policy):
        self.state[self.end] = state
        self.action[self.end] = action
        self.value[self.end] = [value]
        self.policy[self.end] = policy
        
        
        self.end = (self.end + 1) % len(self.state)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.state)
        
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.state) - self.start
        else:
            return self.end - self.start
#        
#    def __iter__(self):
#        for i in range(len(self)):
#            yield self[i]
#    
    def get_data(self):
        
        data_len = self.__len__()
        
        state = np.array(self.state[0 : data_len])
        action = np.array(self.action[0 : data_len])
        value = np.array(self.value[0 : data_len])
        policy = np.array(self.policy[0 : data_len])
        
        
        return state, action, value, policy
    


