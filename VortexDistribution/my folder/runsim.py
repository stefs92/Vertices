# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 03:50:00 2019

@author: stefs
"""
import numpy as np
import vortex_dist as vd
import plotstuff as ps

def runsim(initial_res = 6, final_res = 20, res_step = 2, LL = 1, ksii = 0.01, mm = 1, omegaa = 1, stencil_sizee=3, epsilonn = 0.1, halo_size = 0.2):
    #rho_fun = lambda x,y,z: ((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)**0.5/halo_size*np.exp(-((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)/halo_size**2)
    rho_fun = lambda x,y,z: np.exp(-((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)/halo_size**2)
    sim = vd.vortex_dist(rho_fun,res = initial_res, L = LL, m = mm, omega = omegaa, stencil_size = stencil_sizee,epsilon = epsilonn, ksi = ksii)
    sim.v_p = np.zeros((sim.res,sim.res,sim.res)) + 0.2*np.random.rand(sim.res,sim.res,sim.res)
    ps.surfplot(sim.rho,sim.res,'Density')
    ps.colorplot(sim.rho,sim.res,'Density')
    sim.solve()
    
    for i in np.arange(initial_res, final_res + res_step, res_step):
        sim.upsample(i)
        sim.solve()
        ps.surfplot(sim.n_mag,sim.res,'Vortices, resolution '+str(i))
        ps.colorplot(sim.n_mag,sim.res,'Vortices, resolution '+str(i))
        ps.surfplot(sim.rho,sim.res,'Density, resolution '+str(i))
        ps.colorplot(sim.rho,sim.res,'Density, resolution '+str(i))
    
    