# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:50:30 2020

@author: jingh
"""

import numpy as np
import copy
import time
from DB import RingBuffer
from VIS import Visualization
#from progress.bar import Bar


class MonteCarloTreeSearch():
    def __init__(self, MaxDepth, MaxSearches, MaxIters, c_puct, atask, robot_net, human_net, switch, tau, data_size):
        assert switch > 0
        
        self.MaxDepth = MaxDepth
        self.MaxIters = MaxIters
        self.MaxSearches = MaxSearches
        
        self.atask = atask  # Assembly tree object 'AssembleTask'
        self.c_puct = c_puct
        self.tau = tau  # Temperature parameter
        
        # Neural network for human and robot agent
        #     Method: fit(), predict()
        self.human_net = human_net
        self.robot_net = robot_net
        
        # Buffer for storing training data
        self.human_data = RingBuffer(data_size)
        self.robot_data = RingBuffer(data_size)
        
        # Switch: training scheme
        #         0: training simultaneously
        #        >0: swtich training agent every 'switch' iterations
        self.switch =switch
        
    
    
    def mixed_iterations(self):
        
        for i in range(self.MaxIters):
            # Let robot search first
            
            print('Iterrations: ', i)
#            self.mixed_simulation(create_video=False)
            for j in range(self.switch):
                self.robot_iterations(mixed=True)
                completion_time, computing_time = self.mixed_simulation(create_video=True)
                assert completion_time != 95, "95 Recached!!!!"
                    
            
            for j in range(self.switch):
                self.human_iterations(mixed=True)
                completion_time, computing_time = self.mixed_simulation(create_video=True)
                assert completion_time != 95, "95 Recached!!!!"
            
#            if i % 5 == 0:
#                self.mixed_simulation(create_video=False)
            
        
    def human_iterations(self, mixed=False):
        
#        bar = Bar('Running Iterations', max=self.MaxIters)
        
        if mixed:
            MaxIters = self.switch
        elif not mixed:
            MaxIters = self.MaxIter
        
        for i in range(MaxIters):
#            print('Iterrations: ', i)
            state = copy.deepcopy(self.atask.tree)
#            bar.next()
            while True:
                
                _, _, state, _, terminal, _ = self.human_search(state)
                
                if terminal:
                    break
            
            state, action_mask, value, policy = self.human_data.get_data()
            self.human_net.fit(state, action_mask, value, policy)
#            self.simulation(create_video=False)    
#        bar.finish()
            
        
    def robot_iterations(self, mixed=False):
        
#        bar = Bar('Running Iterations', max=self.MaxIters)
        
        if mixed:
            MaxIters = self.switch
        elif not mixed:
            MaxIters = self.MaxIters
        
        for i in range(MaxIters):
#            print('Iterrations: ', i)
            state = copy.deepcopy(self.atask.tree)
#            bar.next()
            while True:
                
                _, _, state, _, terminal, _ = self.robot_search(state)
                
                if terminal:
                    break
            
            
            state, action_mask, value, policy = self.robot_data.get_data()
            self.robot_net.fit(state, action_mask, value, policy)
#            self.simulation(create_video=False)    
#        bar.finish()
            

    def create_edge(self, parent, action, P, robot):
        
        # Create the edge structure
        #     'parent'  Parent state
        #     'next state'  Next state after take the action
        #     'action'
        #     'N'  Visit count
        #     'W'  Sum of values over searches
        #     'Q'  State-action value
        #     'P'  Prior probability to choose the action
        
        edge = {'parent': parent,
                'child': [],
                'action': action,
                'N': 0,
                'W': 0,
                'Q': 0,
                'P': P}
        
        return edge
        
        
    def create_node(self, state, robot):
        s = copy.deepcopy(state)
        
        alist = self.atask.available_action(state, robot)
        
        if robot:
            _, policy = self.robot_net.predict(state, alist)
        elif not robot:
            _, policy = self.human_net.predict(state, alist)
        
        edges = [self.create_edge(s, x, policy[x], robot) for x in alist]
        
        # Create the node structure
        new_node = {'state': s,
                    'actions': alist,
                    'edge': edges}
        
        
        return new_node
    
    
    def human_search(self, root, test_time=False):
        # This is a human search
        robot = False
        
        root, _, _ = self.let_robot_do(root, False)
    
        # Create root node
        root_node = self.create_node(root, robot)
        
        # Node pool to include all the nodes during searches
        node_pool = [root_node]
        
        for i in range(self.MaxSearches):
            depth = 0
            # Create three list for search pathes
            elapsed_time_path = []
            node_path, edge_path = [0], []
            # Every search begins from the root node
            node_id = 0
            
            while depth <= self.MaxDepth:
                # Get the node to be expanded
                node = copy.deepcopy(node_pool[node_id])
                # Add current node id to the path container
                
                elapsed_time_step = 0
                
                # The Q, N, P for all the edges
                # under current node
                Q, N, P = [], [], []

                for edge in node['edge']:
                    Q.append(edge['Q'])
                    N.append(edge['N'])
                    P.append(edge['P'])
                
                # Select the action per PUCT algorithm
                action = self.puct(Q, N, P, node['actions'])
                
                # Excecute the selected action
                # Return the subsequent state and elapsed time
                state, terminal, elapsed_time = self.atask.next_state(node['state'], node['actions'][action], robot)
                # Add the elapsed time for robot action to
                # the elapsed time during this step
                elapsed_time_step += elapsed_time
                # See if human has something to do or not
                state, terminal, elapsed_time_robot = self.let_robot_do(state, terminal)
                # If human do an action, the time will be
                # added to elapsed_time_step
                elapsed_time_step += elapsed_time_robot
                
                # Add current edge to the edge container
                edge_path.append(action)
                
                # Record the elapsed time during this step
                elapsed_time_path.append(elapsed_time_step)
                
                # See if the subsequent state has been visited or not
                visited = False
                
                for i in range(len(node_pool)):
                    # See if the new node is already in the node pool
                    if (node_pool[i]['state'] == state).all():
                        node_id = i
                        visited = True
                        break
                    
                if not visited:
                    # This node is not visited before
                    # Create a new node
                    new_node = self.create_node(state, robot)
                    # New node will be attached to the end
                    node_id = len(node_pool)
                    node_pool.append(new_node)
                
                node_path.append(node_id)
                
                # Increment depth and continue
                if terminal:
                    break
                elif not terminal:
                    depth += 1
                
            # Now that the search has reached a leaf node
            # We need to backup
            
            if terminal:
                # If the leaf node is terminal, the node
                # has no value, since the work is completed
                value = 0
            elif not terminal:
                # If the leaf node is not terminal, then
                # consult the robot net to get value
                value, _ = self.human_net.predict(state)
                        
            elapsed_time_backup = value
            
            
            
            for i in range(len(node_path)-2, -1, -1):
                # Backup starts from above the leaf node
                # Note: we don't update leaf node
                
                node_id = node_path[i]
                edge_id = edge_path[i]
                # Update accumulated info for edge
                # Visit count add 1
                node_pool[node_id]['edge'][edge_id]['N'] += 1
                # Vaule add elapsed_time_backup
                node_pool[node_id]['edge'][edge_id]['W'] += elapsed_time_backup
                # Update state-action value for edges
                node_pool[node_id]['edge'][edge_id]['Q'] =\
                node_pool[node_id]['edge'][edge_id]['W']/\
                node_pool[node_id]['edge'][edge_id]['N']
                
                elapsed_time_backup += elapsed_time_path[i]
                
        # Now that the maximum searches have been reached
        # Processing the root node
        state = node_pool[0]['state']
        alist = node_pool[0]['actions']
        N = np.zeros(len(alist))
        Q = np.zeros(len(alist))
        for i in range(len(alist)):
            N[i] = node_pool[0]['edge'][i]['N']
            Q[i] = node_pool[0]['edge'][i]['Q']
#        print(N)
        
        if test_time:
            # For test time, we choose the most visited edge
            action = alist[N.argmax()]
            
        elif not test_time:
            pi = np.zeros(len(alist))
            for i in range(len(alist)):
                pi[i] = pow(N[i], 1/self.tau) / pow(N, 1/self.tau).sum()

            value = min(Q)
            
            # For train time, we choose action using temperature param
            action = np.random.choice(alist, p=pi)
#            print(pi)
        # Find the subsequent state after the action
        state, terminal, elapsed_time = self.atask.next_state(state, action, robot)
        
        state_1 = copy.deepcopy(state)
        
        human_next_state, terminal, elapsed_time_robot = self.let_robot_do(state, terminal)
        
        human_elapsed_time = elapsed_time_robot + elapsed_time
        
        
        if not test_time:
            # Fit the neural network, if not test_time
            
            policy = np.zeros(self.atask.w)
            policy[alist] = pi
            action_mask = np.zeros(self.atask.w)
            action_mask[alist] = 1
            
            self.human_data.append(root, action_mask, value, policy)
        
        return state_1, elapsed_time, human_next_state, human_elapsed_time, terminal, action
    
    
    def robot_search(self, root, test_time=False):
        # This is a robot search
        robot = True
        # Create root node
        root_node = self.create_node(root, robot)
        # Node pool to include all the nodes during searches
        node_pool = [root_node]
        
        for i in range(self.MaxSearches):
            depth = 0
            # Create three list for search pathes
            elapsed_time_path = []
            node_path, edge_path = [0], []
            # Every search begins from the root node
            node_id = 0
            
            while depth <= self.MaxDepth:
                # Get the node to be expanded
                node = copy.deepcopy(node_pool[node_id])
                # Add current node id to the path container
                
                elapsed_time_step = 0
                
                # The Q, N, P for all the edges
                # under current node
                Q, N, P = [], [], []

                for edge in node['edge']:
                    Q.append(edge['Q'])
                    N.append(edge['N'])
                    P.append(edge['P'])
                
                # Select the action per PUCT algorithm
                action = self.puct(Q, N, P, node['actions'])
                
                # Excecute the selected action
                # Return the subsequent state and elapsed time
                state, terminal, elapsed_time =\
                self.atask.next_state(node['state'], node['actions'][action], robot)
                # Add the elapsed time for robot action to
                # the elapsed time during this step
                elapsed_time_step += elapsed_time
                # See if human has something to do or not
                state, terminal, elapsed_time_human = self.let_human_do(state, terminal)
                # If human do an action, the time will be
                # added to elapsed_time_step
                elapsed_time_step += elapsed_time_human
                
                
                # Add current edge to the edge container
                edge_path.append(action)
                
                # Record the elapsed time during this step
                elapsed_time_path.append(elapsed_time_step)
                
                
                # See if the subsequent state has been visited or not
                visited = False
                
                for i in range(len(node_pool)):
                    # See if the new node is already in the node pool
                    if (node_pool[i]['state'] == state).all():
                        node_id = i
                        visited = True
                        break
                    
                if not visited:
                    # This node is not visited before
                    # Create a new node
                    new_node = self.create_node(state, robot)
                    # New node will be attached to the end
                    node_id = len(node_pool)
                    node_pool.append(new_node)
                
                node_path.append(node_id)
                
                # Increment depth and continue
                if terminal:
                    break
                elif not terminal:
                    depth += 1
                
            # Now that the search has reached a leaf node
            # We need to backup
            
            if terminal:
                # If the leaf node is terminal, the node
                # has no value, since the work is completed
                value = 0
            elif not terminal:
                # If the leaf node is not terminal, then
                # consult the robot net to get value
                value, _ = self.robot_net.predict(state)
                        
            elapsed_time_backup = value
            
            
            
            for i in range(len(node_path)-2, -1, -1):
                # Backup starts from above the leaf node
                # Note: we don't update leaf node
                
                node_id = node_path[i]
                edge_id = edge_path[i]
                # Update accumulated info for edge
                # Visit count add 1
                node_pool[node_id]['edge'][edge_id]['N'] += 1
                # Vaule add elapsed_time_backup
                node_pool[node_id]['edge'][edge_id]['W'] +=\
                elapsed_time_backup
                # Update state-action value for edges
                node_pool[node_id]['edge'][edge_id]['Q'] =\
                node_pool[node_id]['edge'][edge_id]['W']/\
                node_pool[node_id]['edge'][edge_id]['N']
                
                elapsed_time_backup += elapsed_time_path[i]
                
        # Now that the maximum searches have been reached
        # Processing the root node
        state = node_pool[0]['state']
        alist = node_pool[0]['actions']
        N = np.zeros(len(alist))
        Q = np.zeros(len(alist))
        for i in range(len(alist)):
            N[i] = node_pool[0]['edge'][i]['N']
            Q[i] = node_pool[0]['edge'][i]['Q']
#        print(N)
        
        if test_time:
            # For test time, we choose the most visited edge
            action = alist[N.argmax()]
            
        elif not test_time:
            pi = np.zeros(len(alist))
            for i in range(len(alist)):
                pi[i] = pow(N[i], 1/self.tau) / pow(N, 1/self.tau).sum()

            value = min(Q)
            
            # For train time, we choose action using temperature param
            action = np.random.choice(alist, p=pi)
#            print(pi)
        # Find the subsequent state after the action
        state, terminal, elapsed_time = self.atask.next_state(state, action, robot)
        
        state_1 = copy.deepcopy(state)
        
        robot_next_state, terminal, elapsed_time_human = self.let_human_do(state, terminal)
        
        robot_elapsed_time = elapsed_time_human + elapsed_time
        
        
        if not test_time:
            # Fit the neural network, if not test_time
            
            policy = np.zeros(self.atask.w)
            policy[alist] = pi
            action_mask = np.zeros(self.atask.w)
            action_mask[alist] = 1
            
            self.robot_data.append(root, action_mask, value, policy)
        
        return state_1, elapsed_time, robot_next_state, robot_elapsed_time, terminal, action
    
    
    def let_human_do(self, state, terminal):
        robot = False
        elapsed_time_human = 0
        
        while True:
            # Loop, in case there are consecutive human actions
            if terminal:
                break
                
            elif not terminal:
                # Let human choose first
                human_alist = self.atask.available_action(state, robot)
                if not human_alist:
                    # If human_alist is empty
                    # Robot continues to choose its next action
                    break
                else:
                    # Else, human_alist is not empty
                    # Consult human policy network to choose an action
                    _, policy = self.human_net.predict(state, human_alist)
                    # Get the chosen human action
                    
                    human_action = human_alist[np.argmax(policy[human_alist])]
                    
                    
                    # Do the action and see the next state
                    state, terminal, elapsed_time = self.atask.next_state(state, human_action, robot)
                    
                    # Add to the elasped time in this step
                    elapsed_time_human += elapsed_time
            
        return state, terminal, elapsed_time_human
    
    
    def let_robot_do(self, state, terminal):
        robot = True
        elapsed_time_robot = 0
        
        while True:
            # Loop, in case there are consecutive human actions
            if terminal:
                break
                
            elif not terminal:
                # Let human choose first
                human_alist = self.atask.available_action(state, not robot)
                robot_alist = self.atask.available_action(state, robot)
                if human_alist:
                    # If human action list is not empty
                    # Go back to let human choose next step
                    break
                else:
                    # Otherwise, we take a look at human actions
                    if robot_alist:
                        # If robot has something to do
                        # Let robot do
                        _, policy = self.robot_net.predict(state, robot_alist)
                        
                        # Get the chosen robot action
                        robot_action = robot_alist[np.argmax(policy[robot_alist])]
                        # Do the action and see the next state
                        state, terminal, elapsed_time = self.atask.next_state(state, robot_action, robot)
                        
                        # Add to the elasped time in this step
                        elapsed_time_robot += elapsed_time
            
        return state, terminal, elapsed_time_robot
    
        
    def puct(self, Q, N, P, alist):
        
        uct = np.zeros(len(alist))
        
        for i in range(len(alist)):
            # Calculate UCT
#            P[i] = 1
            uct[i] = Q[i] - self.c_puct * P[i] * np.sqrt(sum(N))/(1+N[i])
            
        # Select the action with largest UCT
        action = np.argmin(uct)
        
        return action
    
    
    def simulation(self, MaxSearches=None, create_video=True):
        # See if we need to adjust MaxSearches during test time
        if create_video:
            self.vis = Visualization()
        
        
        start_time = time.time()
        if MaxSearches is not None:
            temp = copy.deepcopy(self.MaxSearches)
            self.MaxSearches = MaxSearches
            
        # Starting from the initial state  
        state = self.atask.tree
        completion_time = 0
        
        step = 0
        
        while True:
            _, _, state, elapsed_time, terminal = self.robot_search(state, test_time=True)
            
            
            completion_time += elapsed_time
            if create_video:
                self.vis.store_image(state, str(step))
            step += 1
            if terminal:
                break
        
        if MaxSearches is not None:
            self.MaxSearches = temp
        
        end_time = time.time()
        
        computing_time = end_time - start_time
        
        print('The Completion Time is: ', completion_time)
        print('The Computing Time is: ', computing_time)
        
        if create_video:
            self.vis.make_video()
        
        return completion_time, computing_time
        
        
    def mixed_simulation(self, MaxSearches=None, create_video=True):
        # See if we need to adjust MaxSearches during test time
        
        
        # In case one wants to change the max search during test time
        if MaxSearches is not None:
            temp = copy.deepcopy(self.MaxSearches)
            self.MaxSearches = MaxSearches
            
        # Starting from the initial state  
        state = copy.deepcopy(self.atask.tree)
        completion_time = 0
        step = 0
        
        if create_video:
            self.vis = Visualization()
            self.make_video(state, completion_time, step, False, create_video)
        
        start_time = time.time()
        
        while True:
            human_alist = self.atask.available_action(state, False)
            if human_alist:
                step += 1
                state_0 = copy.deepcopy(state)
                
                state, elapsed_time, _, _, terminal, action = self.human_search(state, test_time=True)
                
                self.make_video(state_0, completion_time, step, False, create_video, action=action)
                
                completion_time += elapsed_time
                
                
                self.make_video(state, completion_time, step, False, create_video)
                
                if self.is_terminal(state):
                    break
            else:
                step += 1
                state_0 = copy.deepcopy(state)
                # Note: a potential bug here
                #
                
                state, elapsed_time, _, _, terminal, action = self.robot_search(state, test_time=True)
                
                self.make_video(state_0, completion_time, step, True, create_video, action=action)
                
                completion_time += elapsed_time
                
                self.make_video(state, completion_time, step, True, create_video)
                
                if self.is_terminal(state):
                    break
            
          
        
        if MaxSearches is not None:
            self.MaxSearches = temp
        
        end_time = time.time()
        
        computing_time = end_time - start_time
        
        print('The Completion Time is: ', completion_time)
        print('The Computing Time is: ', computing_time)
        
        if create_video:
            self.vis.make_video()
        
        return completion_time, computing_time
    
    
    def make_video(self, state, completion_time, step, robot, create_video, action=None):
        
        if create_video:
            
            title = 'Step: ' + str(step) +' || Elapsed Time: ' + str(completion_time)
            
            self.vis.store_image(state, title, action, robot, store=True)
            
    
    def is_terminal(self, state):
        terminal = (state == -2).all()
        
        return terminal

    






