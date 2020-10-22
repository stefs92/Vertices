# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:38:46 2018

@author: stefs
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def power_spectrum(psi,dx):
    
    res = psi.shape[0]
    def gradient_squared(field,dx):
        forwardx = np.roll(field,1,axis=0)
        backwardx = np.roll(field,-1,axis=0)
        derivativex = (forwardx - backwardx)/(2.0*dx)
        
        forwardy = np.roll(field,1,axis=0)
        backwardy = np.roll(field,-1,axis=0)
        derivativey = (forwardy - backwardy)/(2.0*dx)
        
        forwardz = np.roll(field,1,axis=0)
        backwardz = np.roll(field,-1,axis=0)
        derivativez = (forwardz - backwardz)/(2.0*dx)
    
        grad_squared = derivativex**2 + derivativey**2 + derivativez**2
        return grad_squared
    
    rho = psi*np.conjugate(psi)
    phase = np.angle(psi)
    energy_density = 0.5*rho*gradient_squared(phase,dx)
    
    power_spec = 1/(res*dx)*np.fft.rfftn(energy_density)
    #plt.imshow(np.mgrid[0:power_spec.shape[1],0:power_spec.shape[2]],power_spec[round(res/2),:,:])
    plt.figure()
    plt.plot(np.arange(power_spec.shape[0]),np.absolute(power_spec[:,1,1]))
    plt.title('Power spectrum in x direction')
    plt.show()
    
    plt.figure()
    plt.plot(np.arange(power_spec.shape[1]),np.absolute(power_spec[1,:,1]))
    plt.title('Power spectrum in y direction')
    plt.show()
    
    plt.figure()
    plt.plot(np.arange(power_spec.shape[2]),np.absolute(power_spec[1,1,:]))
    plt.title('Power spectrum in z direction')
    plt.show()
    return power_spec