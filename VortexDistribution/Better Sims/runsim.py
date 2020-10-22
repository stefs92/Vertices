# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 03:50:00 2019

@author: stefs
"""
import numpy as np
import vortexes_rectangular as vd
import plotstuff as ps

def runsim(initial_res = 8, final_res = 20, res_step = 2,halo_size = 0.3, W = 1.0, L = 5.0, l = 1.0):
    #rho_fun = lambda x,y,z: ((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)**0.5/halo_size*np.exp(-((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)/halo_size**2)
    rho_fun = lambda x,y: 1000*np.exp(-(x**2 + y**2)/halo_size**2)*np.sqrt(x**2 + y**2)**l
    #rho_fun = lambda x,y: np.exp(-(x**2 + y**2)/halo_size**2)
    sim = vd.vortex_dist(rho_fun,res = initial_res, W = W, L = L)
    #sim.v_phi[0:sim.res-1,0:sim.res-1] = np.zeros((sim.res-1,sim.res-1)) + 0.2*np.random.rand(sim.res-1,sim.res-1)
    
    
    #for i in range(initial_res):
    #    for j in range(initial_res):
    #        sim.v_phi[i,j] = np.abs(i - sim.res/2)
    
    
            
    sim.get_curl()
    sim.get_n()
    sim.get_n_mag()
    #ps.surfplot(np.transpose(sim.rho),sim.res,'Density')
    ps.colorplot(sim.rho,sim.res,'Density',sim.L)
    ps.colorplot(sim.n_mag,sim.res,'Vortex Density',sim.L)
    print('v_phi before solving')
    print(sim.v_phi)
    print('quivers before solving')
    ps.quivers(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Quivers',sim.L)
    E = sim.get_energy()
    print('Energy before solving:')
    print(E)
    print(np.transpose(sim.n_mag).shape)
    
    sim.solve()
    #print('vortex density')
    #print(sim.n_mag)
    print('v_phi after first iteration')
    print(sim.v_phi)
    ps.colorplot(sim.n_mag,sim.res,'Vortices, resolution '+str(initial_res),sim.L)
    #ps.surfplot(sim.rho[1:sim.res-1,1:sim.res-1],sim.res-1,'Density, resolution '+str(initial_res))
    ps.colorplot(sim.rho,sim.res,'Density, resolution '+str(initial_res),sim.L)
    ps.quivers(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Quivers',sim.L)
    ps.streams(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Streams',sim.L)
    #print('n_v')
    #print(sim.n_v)
    
    for i in range(0):
        sim.doubl()
        ps.colorplot(sim.n_mag,sim.res,'Vortices after doubling, resolution '+str(sim.res),sim.L)
        sim.solve()
        #ps.surfplot(sim.n_mag[1:sim.res-1,1:sim.res-1],sim.res-1,'Vortices, resolution '+str(i))
        #print('vortex density')
        #print(sim.n_mag)
        print('v_phi')
        print(sim.v_phi)
        ps.colorplot(sim.n_mag,sim.res,'Vortices, resolution '+str(i),sim.L)
        #ps.surfplot(sim.rho[1:sim.res-1,1:sim.res-1],sim.res-1,'Density, resolution '+str(i))
        ps.colorplot(sim.rho,sim.res,'Density, resolution '+str(i),sim.L)
        ps.quivers(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Quivers',sim.L)
        ps.streams(sim.n_v[:,:,1],sim.n_v[:,:,0],sim.res,'Streams',sim.L)
        #print('n_v')
        #print(sim.n_v)
        
    np.savetxt('vortex res',sim.n_mag)
    #ps.volumeplot(sim.n_mag,'vortices')
    
    