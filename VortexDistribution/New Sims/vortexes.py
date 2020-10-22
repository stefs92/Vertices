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
    def __init__(self, rho_func, res = 6, W = 1, L = 1,stencil_size=3, epsilon = .1):
        self.W = W
        self.L = L
        #self.vortex = np.zeros((res, res, res, 3)) + np.random.rand(res,res,res,3) #grid of 3-vectors

       
        #if res%(stencil_size*2 + 1) != 0:
        #    res += stencil_size*2 + 1 - res%(stencil_size*2 + 1)
        self.res = res
        self.epsilon = epsilon
        self.fft_space_potential = np.zeros((self.res,self.res), dtype=np.complex128)
        self.fft_space_phi = np.zeros((self.res,self.res), dtype=np.complex128)
        self.allocate_arrays()
        self.rho_func = rho_func
        self.eval_rho()

    def eval_rho(self):
            # The fcc grid's axes are along the diagonal of each cube's face
            # This has a few advantages, mostly better stencils
        #x = np.linspace(-1, 1, self.res + 1)[:-1]
        #for i in np.ndindex((self.res, self.res)):
#            print(i)
        #    self.rho[i] = self.rho_func(x[i[0]], x[i[1]])
        zsp = np.linspace(-1, 1, 2*self.res + 1)[:-1]
        rsp = np.linspace(0, 1, self.res + 1)[:-1]
        for i in range(self.res):
            for j in range(self.res):
                self.rho[i,j] = self.rho_func(zsp[i], rsp[j])        

    
    def allocate_arrays(self):
        self.v_potential = np.zeros((2*self.res, self.res))
        self.v_phi = np.zeros((2*self.res,self.res))
        self.v_curl = np.zeros((2*self.res,self.res,2))
        self.v_z = np.zeros((2*self.res,self.res))
        self.v_r = np.zeros((2*self.res,self.res))
        self.tmp_whtvr = np.zeros((2*self.res,self.res))
        self.n_v = np.zeros((2*self.res,self.res,2))
        self.n_mag = np.zeros((2*self.res,self.res))
        self.rho = np.zeros((2*self.res,self.res))
        self.v_mag = np.zeros((2*self.res,self.res))

        
    def get_vcomps(self):
        self.v_z = np.array(np.gradient(self.v_potential,axis = 0))
        self.v_r = np.array(np.gradient(self.v_potential,axis = 1))
        
    def get_curl(self):
        # first, get the r - component stored in 1
        self.v_curl[:,:,1] = -1.0*self.res/self.L*np.array(np.gradient(self.v_phi,axis = 0))
        #next, get rho*v_phi
        self.tmp_whtvr[...] = self.v_phi[...]
        for j in range(1,self.res):
            self.tmp_whtvr[:,j] *= j
        # j -> r + 1 everywhere
        
        # differentiate by rho (two factors of res cancel out)
        self.v_curl[:,:,0] = np.array(np.gradient(self.tmp_whtvr,axis = 1))
        
        
        #then, divide by rho
        
        for j in range(1,self.res):
            self.v_curl[:,j,0] /= (j*self.res/self.L)
        
        self.v_curl[:,0,0] = 0
                
        
    def get_n(self):
        self.n_v = 0.5*self.v_curl
        self.n_v[:,:,0] += self.W
        
    def get_n_mag(self):
        self.n_mag = np.sqrt(self.n_v[:,:,0]**2 + self.n_v[:,:,1]**2)
                    
    def get_v_mag(self):
        self.v_mag = np.sqrt(self.v_z**2 + self.v_r**2 + self.v_phi**2)
        
    def get_v_hat_potential(self):
        self.fft_space_potential[...] = fftn(self.v_potential, axes=(0,1))

    def get_v_potential(self):
        self.v_potential[...] = np.real(ifftn(self.fft_space_potential, axes=(0, 1)))
        
    def get_v_hat_phi(self):
        self.fft_space_phi[...] = fftn(self.v_phi, axes=(0,1))

    def get_v_phi(self):
        self.v_phi[...] = np.real(ifftn(self.fft_space_phi, axes=(0, 1)))
        
    
    def get_energy(self):
        E = 0
        self.get_vcomps()
        self.get_curl()
        self.get_n()
        self.get_n_mag()
        self.get_v_mag()
        
        tmpe = 0
        for i in range(2*self.res):
            for j in range(self.res):
                tmpe = 0.25*self.v_mag[i,j]**2
                tmpe += self.n_mag[i,j]**2
                tmpe -= self.n_mag[i,j]
                #tmpe -= 2*np.log(self.n_mag[i,j])*self.n_mag[i,j]**2/np.abs(2*self.W + self.v_curl[i,j,0])*2*(-1.0*np.heaviside(self.n_mag[i,j],0.5)+0.5)
                tmpe -= 2*np.log(self.n_mag[i,j])*self.n_mag[i,j]**2/np.abs(2*self.W + self.v_curl[i,j,0])
                E += tmpe*self.rho[i,j]*(j+1) 
                if self.n_mag[i,j] > 1:
                    E += 100000.0*(self.n_mag[i,j]-1)
                    
        E *= self.L**3/self.res**3
        return E
    
    def solve(self, method = 'CG'):
        def obj_func(v):
          self.v_phi[1:2*self.res-1,0:self.res-1].flat[...] = v[...]
          E = self.get_energy()
          # print(E)
          return E

        A = minimize(obj_func, self.v_phi[1:2*self.res-1,0:self.res-1].flat[...], method=method, jac=False)
        self.v_phi[1:2*self.res-1,0:self.res-1].flat[...] = A.x[...]
        return A
                
    def upsample(self, new_res):
        self.get_v_hat_potential()
        new_fft_space_potential = np.zeros((new_res, new_res), dtype=np.complex128)
        slice_tup = ( slice(None, self.res//2 + 1), slice( -(self.res)//2, None) )
        for i in np.ndindex((2,2)):
            ind = tuple([slice_tup[j] for j in i])
            new_fft_space_potential[ind] = self.fft_space_potential[ind]
        if self.res%2 == 0:
            new_fft_space_potential[self.res//2] /= 2
            new_fft_space_potential[:, self.res//2] /= 2
            new_fft_space_potential[-self.res//2] /= 2
            new_fft_space_potential[:, -self.res//2] /= 2
            
        self.get_v_hat_phi()
        new_fft_space_phi = np.zeros((new_res, new_res), dtype=np.complex128)
        slice_tup = ( slice(None, self.res//2 + 1), slice( -(self.res)//2, None) )
        for i in np.ndindex((2,2)):
            ind = tuple([slice_tup[j] for j in i])
            new_fft_space_phi[ind] = self.fft_space_phi[ind]
        if self.res%2 == 0:
            new_fft_space_phi[self.res//2] /= 2
            new_fft_space_phi[:, self.res//2] /= 2
            new_fft_space_phi[-self.res//2] /= 2
            new_fft_space_phi[:, -self.res//2] /= 2

        self.fft_space_potential = new_fft_space_potential*(new_res**2)/(self.res**2)
        self.fft_space_phi = new_fft_space_phi*(new_res**2)/(self.res**2)
        self.res = new_res
        self.allocate_arrays()
        self.get_v_potential()
        self.get_v_phi()
        self.eval_rho()
