# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:48:14 2018

@author: stefs
"""

def power_spectrum2(psi,length,m=1.0e-22):
    
    hbar = 6.582119514e-16
    
    def ksq_multiplication(field,maxk):
        size = field.shape
        for i in range(0,size[0]):
            for j in range(0,size[1]):
                for l in range(0,size[2]):
                    ksq = (float(i)/size[0]*maxk)**2 + (float(j)/size[1]*maxk)**2 + (float(l)/size[2]*maxk)**2
                    field[i,j,l] = field[i,j,l]*ksq
        return field
    
    res = psi.shape[0]
    dx = length / res
    rho = psi*np.conjugate(psi)
    
    mom_psi = np.fft.fftn(psi)
    total_kinetic = ksq_multiplication(mom_psi*np.conjugate(mom_psi),2*np.pi/dx)*hbar**2/(2.0*m)
    quantum_potential = ksq_multiplication(np.fft.fftn(np.sqrt(rho))**2,2*np.pi/dx)*hbar**2/(2.0*m)
    
    power_spec = total_kinetic - quantum_potential
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