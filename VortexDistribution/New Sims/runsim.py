# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 03:50:00 2019

@author: stefs
"""
import numpy as np
import vortexes as vd
import plotstuff as ps

def runsim(initial_res = 4, final_res = 20, res_step = 2,halo_size = 0.3, W = 1.0, L = 5.0, l = 1.0):
    #rho_fun = lambda x,y,z: ((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)**0.5/halo_size*np.exp(-((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)/halo_size**2)
    rho_fun = lambda x,y: np.exp(-(x**2 + y**2)/halo_size**2)*np.sqrt(x**2 + y**2)**l
    #rho_fun = lambda x,y: np.exp(-(x**2 + y**2)/halo_size**2)
    sim = vd.vortex_dist(rho_fun,res = initial_res, W = W, L = L)
    #sim.v_phi = np.zeros((sim.res,sim.res)) + 10*np.random.rand(sim.res,sim.res)
    
    
    for i in range(1,initial_res-1):
        for j in range(1,initial_res-1):
            sim.v_phi[i,j] = i
            #sim.v_phi[i,j] = 0
            
    sim.get_curl()
    sim.get_n()
    sim.get_n_mag()
    ps.surfplot(sim.rho,sim.res,'Density')
    ps.colorplot(sim.rho,sim.res,'Density')
    ps.colorplot(sim.n_mag,sim.res,'Vortex Density')
    print('v_phi before solving')
    print(sim.v_phi)
    print('quivers before solving')
    ps.quivers(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Quivers')
    E = sim.get_energy()
    print('Energy before solving:')
    print(E)
    
    sim.solve()
    print('Energy after solving:')
    E = sim.get_energy()
    print(E)
    #print('vortex density')
    #print(sim.n_mag)
    print('v_phi after first iteration')
    print(sim.v_phi)
    ps.colorplot(np.transpose(sim.n_mag),sim.res,'Vortices, resolution '+str(initial_res))
    #ps.surfplot(sim.rho[1:sim.res-1,1:sim.res-1],sim.res-1,'Density, resolution '+str(initial_res))
    ps.colorplot(np.transpose(sim.rho),sim.res,'Density, resolution '+str(initial_res))
    ps.quivers(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Quivers')
    ps.streams(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Streams')
    #print('n_v')
    #print(sim.n_v)
    
    for i in (8,12,20,40,80):
        sim.upsample(i)
        sim.solve()
        E = sim.get_energy()
        print(E)
        #ps.surfplot(sim.n_mag[1:sim.res-1,1:sim.res-1],sim.res-1,'Vortices, resolution '+str(i))
        #print('vortex density')
        #print(sim.n_mag)
        print('v_phi')
        print(sim.v_phi)
        ps.colorplot(np.transpose(sim.n_mag),sim.res,'Vortices, resolution '+str(i))
        #ps.surfplot(sim.rho[1:sim.res-1,1:sim.res-1],sim.res-1,'Density, resolution '+str(i))
        ps.colorplot(np.transpose(sim.rho),sim.res,'Density, resolution '+str(i))
        ps.quivers(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Quivers')
        ps.streams(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Streams')

        #print('n_v')
        #print(sim.n_v)
        
    np.savetxt('vortex res',sim.n_mag)
    #ps.volumeplot(sim.n_mag,'vortices')
    
    