# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 05:00:32 2018

@author: stefs
"""

def make_vortices(length_scale = 0.1, ang_momentum = 2.0):
    @jit
    def vortex(x,y,z,xcenter, ycenter):
        x = x - xcenter
        y = y - ycenter
        
        r = np.sqrt(x**2 + y**2)
        rho = np.tanh(r/length_scale)**ang_momentum
        return rho*np.exp(1j*ang_momentum*np.angle(x+ 1j*y))
   
#    def two_vortices(x,y,z):
#        if x < 0.5:
#            return vortex(x,y,z, 0.25, 0.50)
#        else:
#            return vortex(x,y,z, 0.75, 0.50)
    
    def make_chaos(x,y,z):
        return vortex(x,y,z,0.2,0.2)*vortex(x,z,y,-0.2,0.2)*vortex(y,z,x,-0.2,-0.2)
    return make_chaos

def stable_cluster(x,y,z):
    r = np.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)
    return (1 - np.tanh((r-0.125)*16))*1e+6

def make_halo(length_scale = 0.1,ang_momentum = 2.0):
    vort = make_vortices(length_scale, ang_momentum)
    def halo(x,y,z):
        return stable_cluster(x,y,z)*vort(x,y,z)
    return halo