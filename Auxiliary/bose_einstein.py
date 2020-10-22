import numpy as np
import numpy.fft as fft
from numba import vectorize, jit

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
	print 'no cuda'


G = 6.67408e-11
h_bar = 4.1356676625e-15

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


@vectorize
def conj_square(a):
	return a*np.conj(a)

def make_donut_func(twists = 1, major_rad = .5, minor_rad = .25):
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


@jit
def ft_inv_laplace(a, fcc=False):
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
				if k > a.shape[0]/2:
					k = a.shape[0] + 1 - k
				if fcc:
					k_sq = 1.5*(i**2 + j**2 + l**2) - i*j - j*l - i*l
				else:
					k_sq = i**2 + j**2 + l**2
				a[i,j,l] = a[i,j,l]/(k_sq)

@jit
def update_position(a, t_step, m, fcc=False):
	k_sq = 1.0
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			for l in range(a.shape[2]):
				if i > a.shape[0]/2:
					i = a.shape[0] + 1 - i
				if j > a.shape[0]/2:
					j = a.shape[0] + 1 - j
				if k > a.shape[0]/2:
					k = a.shape[0] + 1 - k
				if fcc:
					k_sq = 1.5*(i**2 + j**2 + l**2) - i*j - j*l - j*k
				else:
					k_sq = i**2 + j**2 + l**2
				a[i,j,l] *= np.exp(-1j*t_step*h_bar*m*k_sq)

class fluid_sim:
	def __init__(self, res, func, length=3.0e22, m=1.0e-22, gpu=False, lam=0, fcc=False):
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
		self.fourier_grid = np.empty((res, res, res/2 + 1), dtype=np.complex128) #for storing fourier transform of rho, to transfer to V
		self.psi_hat = np.empty((res,res,res), dtype=np.complex128)
		self.N = float(res**3)

		if gpu:
			self.g_psi = gpuarray.to_gpu(self.psi)
			self.g_rho = gpuarray.to_gpu(self.rho)
			self.g_V = gpuarray.to_gpu(self.V)
			self.g_psi_hat = gpuarray.to_gpu(self.psi_hat)
			self.g_fourier = gpuarray.to_gpu(self.fourier_grid)

			self.psi_plan = cufft.cufftPlan3d(self.psi.shape[0], self.psi.shape[1], self.psi.shape[2], cufft.CUFFT_Z2Z)
			self.rho_plan = cufft.cufftPlan3d(self.rho.shape[0], self.rho.shape[1], self.rho.shape[2], cufft.CUFFT_D2Z)
			self.inv_plan = cufft.cufftPlan3d(self.fourier_grid.shape[0], self.fourier_grid.shape[1], self.fourier_grid.shape[2], cufft.CUFFT_Z2D)

			self.mod = SourceModule(pot_code)
			self.g_conj_square = ElementwiseKernel("pycuda::complex<double>* a, double* b", "b[i] = a[i].real()*a[i].real() + a[i].imag()*a[i].imag()", "conj_square")

			if not fcc:
				self.g_pot_func = self.mod.get_function("square_pot")
				self.g_mom_func = self.mod.get_function("mom_update")

			else:
				self.g_pot_func = self.mod.get_function("fcc_pot")
				self.g_mom_func = self.mod.get_function("fcc_mom_update")

	#def get_g_V(self):
	#	self.g_rho[...] = self.g_psi[...]**2
	#	self.g_fourier_grid[...] = 


	def get_V(self):
		# Get gravitational potential and add rho**2 term
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
		if not self.gpu:
			self.psi *= np.exp(-1j*factor*self.m*self.V)
		else:
			self.g_psi_hat[...] = -1.0j*factor*self.m*self.g_V
			cumath.exp(self.g_psi_hat, out=self.g_psi_hat)
			self.g_psi *= self.g_psi_hat

	def update_position(self, time_step):
		#evolve wave by time_step
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
		self.get_V()
		self.update_momentum(t_step/2)
		for i in range(steps):
			self.update_position(t_step)
			self.get_V()