import numpy as np
import scipy as sp
import numpy.linalg as lin
from numba import jit, vectorize
from numba import int32
from scipy.fftpack import fftn, ifftn
from scipy.optimize import minimize
import scipy.sparse as spar
from scipy.linalg import null_space

#h_bar = 1.055e-34
h_bar = 1.0

class vortex_dist:
    def __init__(self, rho_func, res = 6, W = 1, L = 1,stencil_size=3, epsilon = 1e-10, maxiters = 100000, etol = 0.001, vtol = 0.001, dt = 00.1):
        self.W = W
        self.L = L
        #self.vortex = np.zeros((res, res, res, 3)) + np.random.rand(res,res,res,3) #grid of 3-vectors
        
        self.maxiters = maxiters
        self.etol = etol
        self.vtol = vtol
        self.dt = dt
       
        #if res%(stencil_size*2 + 1) != 0:
        #    res += stencil_size*2 + 1 - res%(stencil_size*2 + 1)
        self.res = res
        self.epsilon = epsilon
        self.tmp_whtvr = np.zeros((2*self.res,self.res), dtype = np.float64)
        self.allocate_arrays()
        self.rho_func = rho_func
        self.eval_rho()

    def eval_rho(self):
            # The fcc grid's axes are along the diagonal of each cube's face
            # This has a few advantages, mostly better stencils
        rsp = np.linspace(0, 1, self.res)
        zsp = np.linspace(-1, 1, 2*self.res)
        for i in np.ndindex((2*self.res, self.res)):
            self.rho[i] = self.rho_func(zsp[i[0]], rsp[i[1]] + 0.5/self.res)     

    
    def allocate_arrays(self):
        self.v_potential = np.zeros((2*self.res,self.res), dtype = np.float64)
        self.v_phi = np.zeros((2*self.res,self.res), dtype = np.float64)
        self.v_curl = np.zeros((2*self.res,self.res,2), dtype = np.float64)
        self.v_z = np.zeros((2*self.res,self.res), dtype = np.float64)
        self.v_r = np.zeros((2*self.res,self.res), dtype = np.float64)
        self.n_v = np.zeros((2*self.res,self.res,2), dtype = np.float64)
        self.n_mag = np.zeros((2*self.res,self.res), dtype = np.float64)
        self.rho = np.zeros((2*self.res,self.res), dtype = np.float64)
        self.v_mag = np.zeros((2*self.res,self.res), dtype = np.float64)
        self.v_gradient = np.zeros((2*self.res-2,self.res-2), dtype = np.float64)

        

        
    def get_curly(self,imin,imax,jmin,jmax):
        # first, get the r - component stored in 1
        self.v_curl[imin:imax,jmin:jmax,1] = -1.0*self.res/self.L*np.array(np.gradient(self.v_phi[imin:imax,jmin:jmax],axis = 0))
        #next, get rho*v_phi
        self.tmp_whtvr[imin:imax,jmin:jmax] = self.v_phi[imin:imax,jmin:jmax]
        for j in range(self.res):
            self.tmp_whtvr[imin:imax,j] *= (j+0.5)*self.L/self.res
        # j -> r + 1 everywhere
        
        # differentiate by rho (two factors of res cancel out)
        self.v_curl[imin:imax,jmin:jmax,0] = np.array(np.gradient(self.tmp_whtvr[imin:imax,jmin:jmax],axis = 1))*self.res/self.L
        
        
        #then, divide by rho
        
        for j in range(self.res):
            self.v_curl[imin:imax,j,0] /= ((j+0.5)*self.L/self.res)
            
        # impose boundary conditions here
        self.v_curl[imin:imax,0,:] = 0
        
    def get_curl(self):
        self.get_curly(0,2*self.res,0,self.res)
        
    def get_ny(self,imin,imax,jmin,jmax):
        self.n_v[imin:imax,jmin:jmax] = 0.5*self.v_curl[imin:imax,jmin:jmax]
        self.n_v[imin:imax,jmin:jmax,0] += self.W
        
    def get_n(self):
        self.get_ny(0,2*self.res,0,self.res)
        
    def get_nmagy(self,imin,imax,jmin,jmax):
        self.n_mag[imin:imax,jmin:jmax] = np.sqrt(self.n_v[imin:imax,jmin:jmax,0]**2 + self.n_v[imin:imax,jmin:jmax,1]**2)
                 
    def get_nmag(self):
        self.get_nmagy(0,2*self.res,0,self.res)
        
    def get_v_hat_phi(self):
        self.fft_space_phi[...] = fftn(self.v_phi, axes=(0,1))

    def get_v_phi(self):
        self.v_phi[...] = np.real(ifftn(self.fft_space_phi, axes=(0, 1)))
        
    
    def get_e(self,imin,imax,jmin,jmax):
        E = 0
        self.get_curly(imin,imax,jmin,jmax)
        self.get_ny(imin,imax,jmin,jmax)
        self.get_nmagy(imin,imax,jmin,jmax)
        
        tmpe = 0
        for i in range(imin,imax):
            for j in range(jmin,jmax):
                tmpe = 0.25*self.v_phi[i,j]**2
                tmpe += self.n_mag[i,j]**2
                tmpe -= self.n_mag[i,j]
                #tmpe -= 2*np.log(self.n_mag[i,j])*self.n_mag[i,j]**2/np.abs(2*self.W + self.v_curl[i,j,0])*2*(-1.0*np.heaviside(self.n_mag[i,j],0.5)+0.5)
                if self.n_mag[i,j] <= 1:
                    tmpe -= 2*np.log(self.n_mag[i,j])*self.n_mag[i,j]**2/np.abs(2*self.W + self.v_curl[i,j,0])
                E += tmpe*self.rho[i,j]*(j + 0.5)
                if self.n_mag[i,j] > 1:
                    E += 100000.0*(self.n_mag[i,j]-1)
                    
        E *= self.L**3/self.res**3
        return E
    
    def get_energy(self):
        return self.get_e(0,2*self.res,0,self.res)
    
    def get_funcd(self,i,j):
        i+=1
        j+=1
        self.v_phi[i,j] += self.epsilon
        Eplus = self.get_e(i-1,i+1,j-1,j+1)
        self.v_phi[i,j] -= 2*self.epsilon
        Eminus = self.get_e(i-1,i+1,j-1,j+1)
        deriv = (Eplus - Eminus)/(2*self.epsilon)
        
        self.v_phi[i,j] += self.epsilon
        
        return deriv
    
    def get_gradient(self):
        for i in range(2*self.res-2):
            for j in range(self.res-2):
                self.v_gradient[i,j] = self.get_lamefd(i,j)
                
    def get_lamefd(self,i,j):
        self.v_phi[i,j] += self.epsilon
        Eplus = self.get_energy()
        self.v_phi[i,j] -= 2*self.epsilon
        Eminus = self.get_energy()
        deriv = (Eplus - Eminus)/(2*self.epsilon)
        self.v_phi[i,j] += self.epsilon
        self.get_curl()
        self.get_n()
        self.get_nmag()
        
        return deriv
    
    def gradient_descend(self):
        oldE = 10000.0
        for i in range(self.maxiters):
            self.get_gradient()
            self.v_phi[1:2*self.res-1,1:self.res-1] -= self.v_gradient[...] * self.dt
            self.get_curl()
            self.get_n()
            self.get_nmag()
            newE = self.get_energy()
            if np.abs(newE - oldE)/self.dt < self.etol and np.max(np.abs(self.v_gradient)) < self.vtol:
                break
            
            if newE > oldE:
                self.dt = self.dt / 2
                print('The time step currently is '+ str(self.dt))
            oldE = newE
            
            if i % 10 == 0:
                print('Just finished '+str(i)+' iterations')
                print('Energy is '+str(newE))
    
    def solve(self, method = 'CG'):
        def obj_func(v):
          self.v_phi[1:2*self.res-1,1:self.res-1].flat[...] = v[...]
          E = self.get_energy()
          self.get_gradient()
          # print(E)
          return E, self.v_gradient.flat[...]

        A = minimize(obj_func, self.v_phi[1:2*self.res-1,1:self.res-1].flat[...], method=method, options = {'disp':True}, jac=True)
        self.v_phi[1:2*self.res-1,1:self.res-1].flat[...] = A.x[...]
        #print(self.get_energy())
        return A
                
    def upsample(self, new_res):

        self.fft_space_potential = new_fft_space_potential*(new_res**2)/(self.res**2)
        self.fft_space_phi = new_fft_space_phi*(new_res**2)/(self.res**2)
        self.res = new_res
        self.allocate_arrays()
        self.get_v_phi()
        self.v_phi[self.res-1,:] = 0
        self.v_phi[:,self.res-1] = 0
        self.eval_rho()
        
    def doubl(self):
        self.tmp_whtvr[...] = self.v_phi[...]
        self.res = 2*self.res
        self.allocate_arrays()
        for i in range(2*self.res):
            for j in range(self.res):
                self.v_phi[i,j] = self.tmp_whtvr[int(i/2),int(j/2)]
                
        self.eval_rho()
        self.tmp_whtvr = np.zeros((2*self.res,self.res))
