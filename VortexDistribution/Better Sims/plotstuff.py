# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:35:32 2018
@author: stefs, rcg
"""


from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def surfplot(psi,res,title,L):
    x = np.linspace(0,L,res)
    y = np.linspace(0,2*L,2*res)
    X,Y = np.meshgrid(x,y)
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot_surface(X,Y,psi[:,:])
    plt.title(title)
    plt.show()

def colorplot(psi,res,title,L):
    plt.figure()
    plt.imshow(psi[...])
    #plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(title)
    plt.xlabel('r')
    plt.ylabel('z')
    plt.show()
    
def quivers(n_x,n_y,res,titl,L):
    #X = np.arange(res)
    #Y = np.arange(res)
    #U, V = np.meshgrid(X, Y)

    fig, ax = plt.subplots()
    q = ax.quiver(n_x,n_y)
    #ax.quiverkey(q, X=0.3, Y=1.1, U=10,
    #label='Quiver key, length = 10', labelpos='E')]
    plt.xlabel('r')
    plt.ylabel('z')
    plt.title(titl)
    plt.show()
    
def streams(n_x,n_y,res,titl,L):
    x = np.linspace(0,L,res)
    y = np.linspace(0,2*L,2*res)
    X,Y = np.meshgrid(x,y)
    plt.xlabel('r')
    plt.ylabel('z')
    #U, V = np.meshgrid(X, Y)

    fig, ax = plt.subplots()
    q = ax.streamplot(X,Y,n_x,n_y)
    #ax.quiverkey(q, X=0.3, Y=1.1, U=10,
    #label='Quiver key, length = 10', labelpos='E')]
    plt.xlabel('r')
    plt.ylabel('z')
    plt.title(titl)
    plt.show()
    
def volumeplot(psi, title):
    mlab.figure(1, bgcolor=(0, 0, 0), size=(350, 350))
    mlab.clf()


    source = mlab.pipeline.scalar_field(psi)
    vol_min = psi.min()
    vol_max = psi.max()
    vol = mlab.pipeline.volume(source, vmin=vol_min + 0.3 * (vol_max - vol_min),
                                   vmax=vol_min + 0.9 * (vol_max - vol_min))

    mlab.view(132, 54, 45, [21, 20, 21.5])

    mlab.show()