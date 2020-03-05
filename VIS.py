# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:01:51 2020

@author: jingh
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import cv2


class Visualization:
    def __init__(self):
        self.task_hatch = ['.', '.', None, None, None]
        self.task_color = [(0.70, 0.70, 0.70),
                           (0.70, 0.70, 0.70),
                           (0.90, 0.40, 0.40),
                           (0.20, 0.82, 0.12),
                           (1.00, 0.95, 0.00)]
        
        self.circle_hatch = [None, None, None, 'x', None]
        self.circle_fcolor = [None, None, 'w', 'w', 'k']
        
        self.filename = []
        
        self.image_idx = 0
        
        if not os.path.exists('./output'):
            os.mkdir('./output')
        
        self.time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        self.fpath = './output/image_' + self.time_stamp
        os.mkdir(self.fpath)
        
    
    def show_image(self, state, title=None, action=None, robot=True):
        self.store_image(state, title, action, robot, False)
    
    
    
    def store_image(self, state, title, action=None, robot=True, store=True):
        
        task_list = []
        
        h = state.shape[0]
        w = state.shape[1]
        
        for row in range(h):
            for col in range(w):
                if state[row, col, 1] >= 0:
                    idx = int(state[row, col, 1])
                    task_list.append([col,
                                      row,
                                      state[row, col, 2],
                                      self.task_hatch[idx],
                                      self.task_color[idx],
                                      self.circle_fcolor[idx],
                                      self.circle_hatch[idx]])
        
        fig, ax = plt.subplots(figsize=(14, 16))
        
        ax.set(xlim=(0, w),
               ylim=(0, h))
        ax.set_title(title, fontsize=24)
        
        """
        task
            task[0]: starting column
            task[1]: starting row
            task[2]: task column to
            task[3]: task hatch style
            task[4]: task facecolor
            task[5]: circle facecolor
            task[6]: circle hatch
        """
        
        for task in task_list:
            rect = patches.Rectangle((task[0], task[1]), task[2], 1, linewidth=2, edgecolor='k', facecolor=task[4], hatch = task[3])
            ax.add_patch(rect)
#            circ = patches.Circle((task[0]+task[2]/2, task[1]+0.5),
#                                  radius=0.3,
#                                  linewidth=2,
#                                  edgecolor='k',
#                                  facecolor=task[5],
#                                  hatch=task[6]
#                                  )
#            ax.add_patch(circ)
            
        # Action gives the position of the action
        if action is not None:
            # We need to annotate the action info
            if robot:
                txt = 'ROBOT'
            else:
                txt = 'HUMAN'
            cols = state[0, action, 2]
            
            ax.text(action+cols/2,
                    0.5,
                    txt,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=24)

                
        
#        ax.text(4, 0.5, 'HERE', horizontalalignment='center', verticalalignment='center', fontsize=25)
        
        if store:
            fname = self.fpath+'/vis'+ str(self.image_idx) +'.png'
            self.image_idx += 1
            plt.savefig(fname=fname)
            self.filename.append(fname)
        
        plt.show()
        
    
    def make_video(self):
        img_array = []

        for fname in self.filename:
            img = cv2.imread(fname)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
     
        fname = './output/video_' + self.time_stamp + '.avi'
        
        out = cv2.VideoWriter(fname,cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
         
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        

if __name__ == '__main__':
    from ATG import at_generator
    tree = at_generator(w=12,
                        h=15,
                        n_human_task=15,
                        n_joint_task=10,
                        n_robot_task=20,
                        max_finish_time=10
                        )
    
    print(tree)
    vis = Visualization()

    vis.show_image(tree, title='HERE')














