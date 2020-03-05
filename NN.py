# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:08:39 2020

@author: jingh
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, multiply

class NeuralNet():
    def __init__(self, h, w, d, batch_size, epoches, value_norm=None):

        self.batch_size = batch_size
        
        self.value_norm = value_norm
        
        self.h = h
        self.w = w
        self.d = d
        
        self.epoches = epoches
        
        # Construct neural network with keras on tensorflow
        
        # Input of normalized states
        state_input = Input(shape=(self.h, self.w, self.d,),
                            name='state')
        # Action mask
        action_input=Input(shape=(self.w,),
                           name='actions')
        
        
        x = Conv2D(filters=10,
                   kernel_size=2,
                   strides=(1, 1),
                   padding='same',
                   activation='relu')(state_input)
        
        x = MaxPooling2D(pool_size=(2, 2))(x)
#        
        x = Conv2D(filters=10,
                   kernel_size=2,
                   strides=(1, 1),
                   padding='same',
                   activation='relu')(x)
        
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Flatten()(x)
        
        x = Dense(units=128,
                  activation='relu')(x)
        
        
        # State value
        value_output = Dense(units=1,
                             activation='linear',
                             name='value')(x)
        # Policy output
        
        x1 = Dense(units=self.w,
                   activation='softmax')(x)
        
        policy_output = multiply([x1, action_input],
                                 name='policy')
        # Complie model
        self.model = Model(inputs=[state_input, action_input],
                           outputs=[value_output, policy_output])
        
        self.model.compile(optimizer='adam',
                           loss={'value': 'mse',
                                 'policy': 'categorical_crossentropy'})


        
    def fit(self, state, action_mask, value, policy):
        
        
#        # Create action mask
#        actions = np.zeros(self.w)
#        actions[alist] = 1
        
        # Augment data in accordance to model requirement
#        state = np.expand_dims(state, axis=0)
#        actions = np.expand_dims(actions, axis=0)
#        value = np.array([[value]])
#        policy = np.expand_dims(policy, axis=0)
        
        # Conduct one step fitting
        
        if self.value_norm is not None:
            value /= self.value_norm
            
        
        self.model.fit(x={'state': state, 'actions': action_mask},
                       y={'value': value, 'policy': policy},
                       epochs=self.epoches,
                       batch_size=self.batch_size,
                       verbose=0)

    def predict(self, state, alist=None):
        
        if alist is None:
            actions = np.ones(self.w)
        else:
            actions = np.zeros(self.w)
            actions[alist] = 1
        
        state = np.expand_dims(state, axis=0)
        actions = np.expand_dims(actions, axis=0)
        
        # Predict value and policy
        value, policy = self.model.predict(x={'state': state, 'actions': actions})
        
        if self.value_norm is not None:
            value *= self.value_norm
        
        return value[0], policy[0]


if __name__ == '__main__':
    """
    Module test
    """
    human_net = NeuralNet(10, 10, 10, 5)
    
    state = np.random.rand(10, 10 ,10)
    alist = [0, 1, 6, 7]
    value = 1.20
    policy = np.zeros(10)
    policy[6] = 1
    
    human_net.fit(state, alist, value, policy)
    
    value, policy = human_net.predict(state)
    print(value, policy)
    
    
    
    
    
    

              
                
                