# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:48:14 2018

@author: stefs
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize, jit
from scipy import stats

tiny = 1e-20

def power_spectrum(psi,length,m=1.0e-22):
    
    if hasattr(psi,'get'):
        psi = psi.get()

    hbar = 6.582119514e-16
    
    @jit
    def add_weight(v,rrho):
        res = np.zeros((v.shape[0],v.shape[1],v.shape[2],v.shape[3]))
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for k in range(v.shape[2]):
                    for l in range(v.shape[3]):
                        res[i,j,k,l] = v[i,j,k,l]*rrho[j,k,l]
        return res
    
    @jit
    def vector_fourier(v):
        comp1 = np.fft.rfftn(v[0,:,:,:])
        comp2 = np.fft.rfftn(v[1,:,:,:])
        comp3 = np.fft.rfftn(v[2,:,:,:])
        res = np.empty((3,comp1.shape[0],comp1.shape[1],comp1.shape[2]),dtype=complex)
        res[0,:,:,:] = comp1
        res[1,:,:,:] = comp2
        res[2,:,:,:] = comp3
        return res
    
    @jit
    def get_incompressible(a,maxk):
        b = np.empty((a.shape[0],a.shape[1],a.shape[2],a.shape[3]),dtype = complex)
        size = a.shape[1]
        for i in range(a.shape[1]):
            i_div = a.shape[1] - i if i > a.shape[1]/2 else i
            ki = i_div*maxk/size
            for j in range(a.shape[2]):
                j_div = a.shape[2] - j if j > a.shape[2]/2 else j
                kj = j_div*maxk/size
                for l in range(a.shape[3]):
                    #l_div = a.shape[3] - l if l > a.shape[3]/2 else l
                    l_div = l
                    kl = l_div*maxk/size
                    kvec = np.array([ki,kj,kl],dtype = float)
                    b[:,i,j,l] = np.cross(kvec,a[:,i,j,l])*(1j)/(ki**2 + kj**2 + kl**2 + tiny)
        return b
    
    @jit
    def integrate_shell(field,maxk):
        field_sq = np.zeros((field.shape[1],field.shape[2],field.shape[3]))
        for i in range(field.shape[1]):
            for j in range(field.shape[2]):
                for k in range(field.shape[3]):
                    field_sq[i,j,k] = np.absolute(field[0,i,j,k])**2 + np.absolute(field[1,i,j,k])**2 + np.absolute(field[2,i,j,k])**2
        size = field.shape[1]
        largestk = int(round(size/2*3**0.5))
        #largestk = size/2
        powers = np.zeros(largestk)
        numtimes = np.zeros(largestk)

        for i in range(size/2):
            for j in range(size/2):
                for k in range(size/2):
                    knorm = (i**2 + j**2 + k**2)**0.5
                    knormint = int(round(knorm))
                    powers[knormint] += np.absolute(field_sq[i,j,k])*knorm**2*maxk**2/size**2
                    numtimes[knormint] += 1
        powers = powers / numtimes
        return powers
        
    #@jit
    #def ksq_multiply(a, maxk, fcc=False):
    #    for i in range(a.shape[0]):
    #        i_div = a.shape[0] - i if i > a.shape[0]/2 else i
    #        for j in range(a.shape[1]):
    #            j_div = a.shape[0] - j if j > a.shape[0]/2 else j
    #            for l in range(a.shape[2]):
    #                l_div = a.shape[0] - l if l > a.shape[0]/2 else l
    #                if fcc:
    #                    k_sq = 1.5*(i_div**2 + j_div**2 + l_div**2) - i_div*j_div - j_div*l_div - i_div*l_div
    #                else:
    #                    k_sq = (i_div**2 + j_div**2 + l_div**2)*maxk**2/a.shape[0]**2
    #                a[i,j,l] *= k_sq
    #    return a
	

    res = psi.shape[0]
    dx = float(length) / float(res)
    maxk = 2*np.pi/dx
    
    sqrtrho = np.real((psi*np.conjugate(psi) + tiny)**0.5)
    exp_phase = psi / (sqrtrho + tiny)
    
    first_term = np.array(np.gradient(sqrtrho,dx,dx,dx))*exp_phase
    second_term = np.array(np.gradient(psi,dx,dx,dx)) - first_term
    velocity = np.real(second_term / (1j*sqrtrho*exp_phase * m / hbar + 1j*tiny))
    #velocity = second_term / (1j*sqrtrho*exp_phase * m / hbar + 1j*tiny)
    
    #weighted_velocity = add_weight(velocity,sqrtrho)
    weighted_velocity = velocity*sqrtrho
    momspace_weighted_velocity = vector_fourier(weighted_velocity)
    weighted_inc_velocity = get_incompressible(momspace_weighted_velocity,maxk)
    powers = integrate_shell(weighted_inc_velocity,maxk)
    
    kvec = np.arange(powers.shape[0])*maxk
    plt.figure()
    #plt.plot(np.log(kvec + tiny),np.log(powers + tiny))
    #plt.plot(kvec,powers)
    maxi = int(np.floor(1/3**0.5*powers.shape[0]))
    plt.loglog(kvec[1:maxi],powers[1:maxi],label = 'spectrum')
    plt.loglog(kvec[5:maxi],powers[5]*kvec[5]**(1.67)*kvec[5:maxi]**(-1.67),label = 'k^(-5/3)')
    plt.loglog(kvec[5:maxi],powers[5]*kvec[5]**(3.0)*kvec[5:maxi]**(-3.0),label = 'k^(-3)')
    #slope,intercept,r_value,p_value,std_err = stats.linregress(np.log(powers + 1e-20),np.log(kvec + 1e-20))
    #print "The power of the power spectrum is", slope, "and the intercept is",intercept
    #plt.plot(kvec,np.e**intercept*kvec**slope)
    plt.xlabel('k (1/kpc)')
    plt.title('Power spectrum')
    plt.legend()
    plt.show()
    
    # maxind = np.argmax(power_spec[1,10:res/2,1])
    #print "maxind", maxind    

    return powers