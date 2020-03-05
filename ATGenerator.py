# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:37:23 2020

@author: jingh
"""

import numpy as np
from VIS import Visualization

def at_generator(w, h, n_human_task, n_robot_task, n_joint_task, max_finish_time, seed=123):
    # Set random seed
    np.random.seed(seed=seed)
    # Num of all tasks
    n_all_task = n_human_task + n_joint_task + n_robot_task
    # Make sure that task counts larger
    # than the chess board height
    assert n_all_task > h
    
    # All possible task locations
    # Note: the first column of every
    # row is already occupied
    task_pos = [(row, col) for row in range(h) for col in range(1, w)]
    
    # The first column of every row must be filled
    task_must = [(row, 0) for row in range(h)]
    
    # Besides task_must
    n_remain_task = n_all_task - h
    
    #Randomly choose remaining task from the positions
    task_id = np.random.choice(len(task_pos), n_remain_task, replace=False)
    
    # All the chosen task positions
    tasks = [task_pos[x] for x in task_id] + task_must
    
    # Shuffle the tasks
    np.random.shuffle(tasks) 

    del task_pos
    
    tree = -np.ones((h, w, 3))
    
    for i in range(n_robot_task):
        idx = i
        # Random finish time
        
        tree[tasks[idx] + (0,)] = np.random.randint(1, high=max_finish_time)
        tree[tasks[idx] + (1,)] = 2
        
    for i in range(n_joint_task):
        idx = i + n_robot_task
        
        tree[tasks[idx] + (0,)] = np.random.randint(1, high=max_finish_time)
        tree[tasks[idx] + (1,)] = 3
    
    for i in range(n_human_task):
        idx = i + n_robot_task + n_joint_task
        
        tree[tasks[idx] + (0,)] = np.random.randint(1, high=max_finish_time)
        tree[tasks[idx] + (1,)] = 4
    
    
    # Determine the d = 2 matrix
    
    for row in range(h):
        for col in range(w):
            if tree[row, col, 1] != -1:
                cols = 1
                if col + cols < w:
                    while tree[row, col+cols, 1] == -1:
                        cols += 1
                        if col + cols >= w:
                            break
                        
                    
                tree[row, col, 2] = cols
    
    
    return tree
        
        
if __name__ == '__main__':
    tree = at_generator(w=12,
                        h=15,
                        n_human_task=15,
                        n_joint_task=10,
                        n_robot_task=20,
                        max_finish_time=10
                        )

    vis = Visualization()

    vis.show_image(tree)
    





