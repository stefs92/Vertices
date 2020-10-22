# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:35:32 2018

@author: stefs
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def plotstuff(psi,res):
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    X,Y = np.meshgrid(x,y)
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot_surface(X,Y,abs(psi)[:,:,50])
    plt.show()

    
    