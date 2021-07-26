import numpy as np
import numpy.fft as fft
from numba import vectorize, jit
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda import cumath
    #from pycuda.tools import make_default_context
    import pycuda.gpuarray as gpuarray
    from pycuda.elementwise import ElementwiseKernel
    from pycuda.compiler import SourceModule
    from skcuda import cufft
except:
    print ('no cuda')

# some physical constants
G = 6.67408e-11
h_bar = 4.1356676625e-15

@vectorize
def conj_square(a):
    # function that computes |a|^2 of the input a
    return a*np.conj(a)
        
# We provide numba versions of key functions here, to be used when a GPU is not available      
@jit
def ft_inv_laplace(a, fcc=False):
    """
    ft_inv_laplace - applies inverse Laplacian in momentum space

    Parameters
    ----------
    a - 3D numpy array that the inverse Laplacian is applied to
    fcc -  boolean value indicating whether to use the face centered cubic grid

    Returns
    -------
    None.
    """
    k_sq = 1.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for l in range(a.shape[2]):
                if (i == 0) and (j == 0) and (l == 0):
                    a[i,j,l] = 0
                    continue
                if i > a.shape[0]/2:
                    i = a.shape[0] + 1 - i
                if j > a.shape[0]/2:
                    j = a.shape[0] + 1 - j
                if l > a.shape[0]/2:
                    l = a.shape[0] + 1 - l
                if fcc:
                    k_sq = 1.5*(i**2 + j**2 + l**2) - i*j - j*l - i*l
                else:
                    k_sq = i**2 + j**2 + l**2
                a[i,j,l] = a[i,j,l]/(k_sq)

@jit
def update_position(a, t_step, m, fcc=False):
    """
    update_position - performs a time step in the momentum space

    Parameters
    ----------
    a - 3D numpy array representing the BEC field
    t_step - the size of the time step
    m - the total mass of the system
    fcc -  boolean value indicating whether to use the face centered cubic grid

    Returns
    -------
    None.
    """
    k_sq = 1.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for l in range(a.shape[2]):
                if i > a.shape[0]/2:
                    i = a.shape[0] + 1 - i
                if j > a.shape[0]/2:
                    j = a.shape[0] + 1 - j
                if l > a.shape[0]/2:
                    l = a.shape[0] + 1 - l
                if fcc:
                    k_sq = 1.5*(i**2 + j**2 + l**2) - i*j - j*l - j*k
                else:
                    k_sq = i**2 + j**2 + l**2
                a[i,j,l] *= np.exp(-1j*t_step*h_bar*m*k_sq)

# Here, we provide several C/C++ methods for use with pycuda
pot_code = '''
#include <cuComplex.h>

__global__ void square_pot(cuDoubleComplex* V, double C, int x_max, int y_max, int z_max)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int i = x*y_max*z_max + y*z_max + z;
    if (i >= x_max*y_max*z_max) return;

    if (x > x_max/2) x = x_max - x;
    if (x > x_max/2) y = x_max - y;
    if (x > x_max/2) z = x_max - z;

    double factor = C/(x*x + y*y + z*z);
    V[i].x = V[i].x*factor;
    V[i].y = V[i].y*factor;
}

__global__ void fcc_pot(cuDoubleComplex* V, double C, int x_max, int y_max, int z_max)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int i = x*y_max*z_max + y*z_max + z;
    if (i >= x_max*y_max*z_max) return;

    if (x > x_max/2) x = x_max - x;
    if (x > x_max/2) y = x_max - y;
    if (x > x_max/2) z = x_max - z;

    double factor = C/(1.5*x*x + 1.5*y*y + 1.5*z*z - x*y - y*z - x*z);
    V[i].x = V[i].x*factor;
    V[i].y = V[i].y*factor;
}

__global__ void mom_update(cuDoubleComplex* V, double C, int x_max, int y_max, int z_max)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int i = x*y_max*z_max + y*z_max + z;
    if (i >= x_max*y_max*z_max) return;

    if (x > x_max/2) x = x_max - x;
    if (x > x_max/2) y = x_max - y;
    if (x > x_max/2) z = x_max - z;

    cuDoubleComplex factor;

    sincos(C*(x*x + y*y + z*z), &factor.y, &factor.x);
    double tmp = V[i].x*factor.x - V[i].y*factor.y;
    V[i].y = V[i].y*factor.x + V[i].x*factor.y;
    V[i].x = tmp;
}

__global__ void fcc_mom_update(cuDoubleComplex* V, double C, int x_max, int y_max, int z_max)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int i = x*y_max*z_max + y*z_max + z;
    if (i >= x_max*y_max*z_max) return;

    if (x > x_max/2) x = x_max - x;
    if (x > x_max/2) y = x_max - y;
    if (x > x_max/2) z = x_max - z;

    cuDoubleComplex factor; //= C*(1.5*x*x + 1.5*y*y + 1.5*z*z - x*y - y*z - x*z);
    double tmp = V[i].x*factor.x - V[i].y*factor.y;
    V[i].y = V[i].y*factor.x + V[i].x*factor.y;
    V[i].x = tmp;
}'''

class fluid_sim:
    """
    fluid_sim: 
        - Class implementing a simulation of Bose-Einstein condensate within a gravitational potential
            
    :instance attributes:
    	grid_dist - spacing between grid points in the simulation
        length - total length of the simulated space
	lam - quartic coupling constant of the Bose-Einstein condensate
	m - total mass
	gpu - boolean value indicating whether to use a GPU or not
	psi - Bose-Einstein field specified by a 3D numpy array
	rho - mass density field specified by a 3D numpy array
	V - gravitational potential spcified by a 3D numpy array
	fourier_grid - attribute for storing fourier transform of rho
	psi_hat - Fourier space representation of psi
	N - number of grid points

        
    :instance methods:
        get_V - get gravitational potential and add rho**2 term
	update_momentum - perform an update in the Fourier space
	update_position - evolve wave by time step
	leap_frog_steps - perform a number of simulation steps
    """

    def __init__(self, res, func, length=3.0e22, m=1.0e-22, gpu=False, lam=0, fcc=False):
	"""
	initializer

        Parameters
        ----------
        res - simulation resolution
	func - function generating initial values for psi
	length - total length of the simulated space
	m - total mass
	gpu - bollean value indicating whether to use a GPU
	lam - quartic coupling constant of the Bose-Einstein condensate
	fcc - boolean value specifying whether to use a face centered cubic grid

	Returns
        -------
        None.

	"""
        self.grid_dist = length/res
        self.lam = lam
        self.m = m
        self.gpu = gpu
        self.psi = np.empty((res,res,res), dtype=np.complex128)
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    self.psi[i,j,k] = func(float(i)/res, float(j)/res, float(k)/res)
        self.rho = np.empty(self.psi.shape, dtype=np.float64)
        self.V = np.empty(self.psi.shape, dtype=np.float64)
        self.fourier_grid = np.empty((res, res, math.floor(res/2 + 1)), dtype=np.complex128) #for storing fourier transform of rho, to transfer to V
        self.psi_hat = np.empty((res,res,res), dtype=np.complex128)
        self.N = float(res**3)

        if gpu:
            # initialize the GPU arrays	    
            self.g_psi = gpuarray.to_gpu(self.psi)
            self.g_rho = gpuarray.to_gpu(self.rho)
            self.g_V = gpuarray.to_gpu(self.V)
            self.g_psi_hat = gpuarray.to_gpu(self.psi_hat)
            self.g_fourier = gpuarray.to_gpu(self.fourier_grid)

            self.psi_plan = cufft.cufftPlan3d(self.psi.shape[0], self.psi.shape[1], self.psi.shape[2], cufft.CUFFT_Z2Z)
            self.rho_plan = cufft.cufftPlan3d(self.rho.shape[0], self.rho.shape[1], self.rho.shape[2], cufft.CUFFT_D2Z)
            self.inv_plan = cufft.cufftPlan3d(self.fourier_grid.shape[0], self.fourier_grid.shape[1], self.fourier_grid.shape[2], cufft.CUFFT_Z2D)
            
            # load the functions provided in pot_code above
            self.mod = SourceModule(pot_code)
            self.g_conj_square = ElementwiseKernel("pycuda::complex<double>* a, double* b", "b[i] = a[i].real()*a[i].real() + a[i].imag()*a[i].imag()", "conj_square")

            if not fcc:
                self.g_pot_func = self.mod.get_function("square_pot")
                self.g_mom_func = self.mod.get_function("mom_update")

            else:
                self.g_pot_func = self.mod.get_function("fcc_pot")
                self.g_mom_func = self.mod.get_function("fcc_mom_update")


    def get_V(self):
	"""
	get_V - computes and stores the gravitational potential

        Parameters
        ----------
 	None.

	Returns
        -------
        None.

	"""
        if not self.gpu:
            self.rho[...] = conj_square(self.psi)
            self.fourier_grid[...] = fft.rfftn(self.rho)
            ft_inv_laplace(self.fourier_grid)
            self.fourier_grid *= 4*np.pi*G
            self.V[...] = fft.irfftn(self.fourier_grid)
            self.V[...] += self.lam*self.rho**2
        else:
            self.g_conj_square(self.g_psi, self.g_rho)
            cufft.cufftExecD2Z(self.rho_plan, self.g_rho.ptr, self.g_fourier.ptr)
            self.g_fourier /= self.psi.shape[0]**3
            self.g_pot_func(self.g_fourier, np.float64(4*np.pi*G/self.N), np.int64(self.fourier_grid.shape[0]), np.int64(self.fourier_grid.shape[1]), np.int64(self.fourier_grid.shape[2]), block=(8,8,8), grid=tuple([(i+7)/8 for i in self.psi_hat.shape]))
            cufft.cufftExecZ2D(self.inv_plan, self.g_fourier.ptr, self.g_V.ptr)
            self.g_V += self.lam*self.g_rho**2

    def update_momentum(self, factor):
	"""
	update_momentum - performs an update in momentum space

        Parameters
        ----------
 	factor - essentially the size of the time step dt

	Returns
        -------
        None.

	"""
        if not self.gpu:
            self.psi *= np.exp(-1j*factor*self.m*self.V)
        else:
            self.g_psi_hat[...] = -1.0j*factor*self.m*self.g_V
            cumath.exp(self.g_psi_hat, out=self.g_psi_hat)
            self.g_psi *= self.g_psi_hat

    def update_position(self, time_step):
        """
	update_momentum - evolves the wave by one time step

        Parameters
        ----------
 	time_step - the size of the time step

	Returns
        -------
        None.

	"""
        if not self.gpu:
            self.psi_hat[...] = fft.fftn(self.psi)
            update_position(self.psi_hat, time_step, self.m)
            self.psi[...] = fft.ifftn(self.psi_hat)
        else:
            cufft.cufftExecZ2Z(self.psi_plan, self.g_psi.ptr, self.g_psi_hat.ptr, cufft.CUFFT_FORWARD)
            self.g_psi_hat /= self.N
            self.g_mom_func(self.g_psi_hat, np.float64(time_step), np.int64(self.psi_hat.shape[0]), np.int64(self.psi_hat.shape[1]), np.int64(self.psi_hat.shape[2]), block=(8,8,8), grid=tuple([(i+7)/8 for i in self.psi_hat.shape]))
            cufft.cufftExecZ2Z(self.psi_plan, self.g_psi_hat.ptr, self.g_psi.ptr, cufft.CUFFT_INVERSE)

    def leap_frog_steps(self, steps, t_step):
        """
	leap_frog_steps - updates the simulation by a number of time steps

        Parameters
        ----------
	steps - the number of time steps to perform
 	t_step - the size of the time step

	Returns
        -------
        None.

	"""

	# first, compute the gravitational potential
        self.get_V()
	# then, give the psi field an initial kick - common in leap frog-type algorithms
        self.update_momentum(t_step/2)
	# then, just simulate a number of steps into the future
        for i in range(steps):
            self.update_position(t_step)
            self.get_V()
            self.update_momentum(t_step)
            print('step ', i)

    def plot_slice(self,res):
        """
	plot_slice - plots a horizontal slice of abs(psi) at the median height

        Parameters
        ----------
	res - the number of grid points along either of the axes

	Returns
        -------
        None.

	"""
        x = np.linspace(0,1,res)
        y = np.linspace(0,1,res)
        X,Y = np.meshgrid(x,y)
        plt.figure()
        ax = plt.axes(projection = '3d')
        ax.plot_surface(X,Y,abs(self.psi)[:,:,math.floor(res/2)])
        plt.show()
        
    def plot_phase(self,res):
        """
	plot_phase - plots a horizontal slice of the phase of phi at the median height

        Parameters
        ----------
	res - the number of grid points along either of the axes

	Returns
        -------
        None.

	"""
        x = np.linspace(0,1,res)
        y = np.linspace(0,1,res)
        X,Y = np.meshgrid(x,y)
        plt.figure()
        ax = plt.axes(projection = '3d')
        ax.plot_surface(X,Y,np.angle(self.psi)[:,:,math.floor(res/2)])
        plt.show()

# The following functions can be useful for creating an initial field configuration
def make_donut_func(twists = 1, major_rad = .5, minor_rad = .25):
    """
    make_donut_func - produces a function for creating a "twisted" donut-shaped initial field configuration

    Parameters
    ----------
    twists - the twist of the field configuration
    major_rad - major radius of the torus (donut)
    minor_rad - minor radius of the torus (donut)

    Returns
    -------
    a function for creating a "twisted" torus configuration
    """
    @jit
    def donut(x, y, z):
        x = (x - .5)*2
        y = (y - .5)*2
        z = (z - .5)*2

        r = np.sqrt(x**2 + y**2)
        r_sq = (r - major_rad)**2
        denom = r_sq + z**2 - minor_rad**2
        if denom >= 0:
            return 0
        else:
            return np.exp(1/denom + 1/(minor_rad**2))*np.exp(np.angle(x + y*1j)*twists*1j)

    return donut

def make_vortices(length_scale = 0.1, ang_momentum = 2.0):
    """
    make_vortices - produces a function for creating vortex-like initial field configurations

    Parameters
    ----------
    length_scale - the length scale of the vortex
    ang_momentum - the angular momentum of the vortex

    Returns
    -------
    a function that creates two vortices for the initial field configuration
    """
    @jit
    def vortex(x,y,z,xcenter, ycenter):
        x = x - xcenter
        y = y - ycenter
        
        r = np.sqrt(x**2 + y**2)
        rho = np.tanh(r/length_scale)**ang_momentum
        return rho*np.exp(1j*ang_momentum*np.angle(x+ 1j*y))
   
    def two_vortices(x,y,z):
        if x < 0.5:
            return vortex(x,y,z, 0.25, 0.50)
        else:
            return vortex(x,y,z, 0.75, 0.50)
    return two_vortices
