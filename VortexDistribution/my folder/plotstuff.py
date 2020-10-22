# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:35:32 2018

@author: stefs
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def surfplot(psi,res,title):
    x = np.linspace(0,1,res)
    y = np.linspace(0,1,res)
    X,Y = np.meshgrid(x,y)
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot_surface(X,Y,psi[:,:,int(res/2)])
    plt.title(title)
    plt.show()

def colorplot(psi,res,title):
    plt.figure()
    plt.imshow(psi[:,:,int(res/2)])
    plt.colorbar()
    plt.title(title)
    plt.show()
    