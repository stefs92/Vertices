import numpy as np
import numpy.fft as fft
from numba import vectorize, jit
from string import Template
import plotstuff as ps

#try:
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import cumath
#from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
from skcuda import cufft
#except:
#print 'no cuda'

#pot_code = open('utilities.cu')


G = 6.67408e-11
#G = 0
h_bar = 6.582119514e-16
#h_bar = 1.0545718e-34 keep it in eV for now
light_speed = 299792458

#ew_pos_update = ElementwiseKernel("pycuda::complex<double>* V, double C",
ew_pos_code = Template('''
//get coordinates from index
// A, B, and C are the array dimensions to be substituted out later.
unsigned int tmp = i;
unsigned int x = tmp/(${B}*${C});
tmp = tmp - x*${B}*${C};
unsigned int y = tmp/${C};
unsigned int z = tmp - y*${C};

if (x > ${A}/2) x = ${A} - x;
if (y > ${B}/2) y = ${B} - y;
if (z > ${C}/2) z = ${C} - z;

//D is an expression defining the wave number
int k_sq = ${D};
pycuda::complex<double> factor(0, -coef*k_sq);

V[i] = V[i]*exp(factor);
''')



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
	for i in range(a.shape[0]):
		i_div = a.shape[0] - i if i > a.shape[0]/2 else i
		for j in range(a.shape[1]):
			j_div = a.shape[0] - j if j > a.shape[0]/2 else j
			for l in range(a.shape[2]):
				l_div = a.shape[0] - l if l > a.shape[0]/2 else l
				if (i == 0) and (j == 0) and (l == 0):
					a[i,j,l] = 0
					continue
				if fcc:
					k_sq = 1.5*(i_div**2 + j_div**2 + l_div**2) - i_div*j_div - j_div*l_div - i_div*l_div
				else:
					k_sq = i_div**2 + j_div**2 + l_div**2
				a[i,j,l] = a[i,j,l]/(k_sq)

@jit
def update_position(a, t_step, m, fcc=False):
	for i in range(a.shape[0]):
		i_div = a.shape[0] - i if i > a.shape[0]/2 else i
		for j in range(a.shape[1]):
			j_div = a.shape[0] - j if j > a.shape[0]/2 else j
			for l in range(a.shape[2]):
				l_div = a.shape[0] - l if l > a.shape[0]/2 else l
				if fcc:
					k_sq = 1.5*(i_div**2 + j_div**2 + l_div**2) - i_div*j_div - j_div*l_div - i_div*l_div
				else:
					k_sq = i_div**2 + j_div**2 + l_div**2
				a[i,j,l] *= np.exp(-1j*t_step*(h_bar/m)*k_sq)

class fluid_sim:
	def __init__(self, res, func, length=0.01, m=1.0e-22, gpu=True, lam=0.01, fcc=False):
		self.grid_dist = length/res
		self.length = length
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
			pot_code = open('utilities.cu')

			self.g_psi = gpuarray.to_gpu(self.psi)
			self.g_rho = gpuarray.to_gpu(self.rho)
			self.g_V = gpuarray.to_gpu(self.V)
			#We save memory by assigning matrices as views of other parts of memory
			self.g_psi_hat = gpuarray.to_gpu(self.psi_hat)
			self.g_fourier = self.g_psi_hat[:res/2 + 1].reshape((res, res, res/2 + 1))

			self.psi_plan = cufft.cufftPlan3d(self.psi.shape[0], self.psi.shape[1], self.psi.shape[2], cufft.CUFFT_Z2Z)
			self.rho_plan = cufft.cufftPlan3d(self.rho.shape[0], self.rho.shape[1], self.rho.shape[2], cufft.CUFFT_D2Z)
			self.inv_plan = cufft.cufftPlan3d(self.rho.shape[0], self.rho.shape[1], self.rho.shape[2], cufft.CUFFT_Z2D)

			self.mod = SourceModule(pot_code.read())
			self.g_conj_square = ElementwiseKernel("pycuda::complex<double>* a, double* b", "b[i] = a[i].real()*a[i].real() + a[i].imag()*a[i].imag()", "conj_square")

			if not fcc:
				self.g_pot_func = self.mod.get_function("square_pot")
				self.g_mom_func = ElementwiseKernel("pycuda::complex<double>* V, double coef",
					ew_pos_code.substitute(A=str(res),B=str(res),C=str(res),D="x*x + y*y + z*z"))

			else:
				self.g_pot_func = self.mod.get_function("fcc_pot")
				self.g_mom_func = ElementwiseKernel("pycuda::complex<double>* V, double coef",
					ew_pos_code.substitute(A=str(res),B=str(res),C=str(res),D="x*x + y*y + z*z"))


	#def get_g_V(self):
	#	self.g_rho[...] = self.g_psi[...]**2
	#	self.g_fourier_grid[...] = 


	def get_V(self):
		# Get gravitational potential and add rho**2 term
		if not self.gpu:
			self.rho[...] = conj_square(self.psi)
			self.fourier_grid[...] = fft.rfftn(self.rho)
			ft_inv_laplace(self.fourier_grid)
			self.fourier_grid *= (self.length**2)*G/np.pi
			self.V[...] = fft.irfftn(self.fourier_grid)
			self.V[...] += self.lam*self.rho**2
		else:
			self.g_conj_square(self.g_psi, self.g_rho)
			cufft.cufftExecD2Z(self.rho_plan, self.g_rho.ptr, self.g_fourier.ptr)
			self.g_pot_func(self.g_fourier, np.float64(G/(np.pi*self.length)), 
				np.int32(self.fourier_grid.shape[0]), 
				np.int32(self.fourier_grid.shape[1]), 
				np.int32(self.fourier_grid.shape[2]), 
				block=(8,8,8), grid=tuple([(i+7)/8 for i in self.fourier_grid.shape]))

			cufft.cufftExecZ2D(self.inv_plan, self.g_fourier.ptr, self.g_V.ptr)
			self.g_V /= self.N
			self.g_V += self.lam*self.g_rho**2

	def update_momentum(self, factor):
		if not self.gpu:
			self.psi *= np.exp(-1j*factor*(self.m/h_bar)*self.V)
		else:
			self.g_psi_hat[...] = -1.0j*factor*(self.m/h_bar)*self.g_V
			cumath.exp(self.g_psi_hat, out=self.g_psi_hat)
			self.g_psi *= self.g_psi_hat

	def update_position(self, time_step):
		#evolve wave by time_step
		if not self.gpu:
			self.psi_hat[...] = fft.fftn(self.psi)
			update_position(self.psi_hat, time_step*(4*np.pi**2)/(self.length**2), self.m)
			self.psi[...] = fft.ifftn(self.psi_hat)
		else:
			cufft.cufftExecZ2Z(self.psi_plan, self.g_psi.ptr, self.g_psi_hat.ptr, cufft.CUFFT_FORWARD)
			copy = self.g_psi_hat.get()
			factor = np.float64(time_step*(h_bar/self.m)*4*(np.pi**2)/(self.length**2) )
			self.g_mom_func(self.g_psi_hat, factor)
			cufft.cufftExecZ2Z(self.psi_plan, self.g_psi_hat.ptr, self.g_psi.ptr, cufft.CUFFT_INVERSE)
			self.g_psi /= self.N

	def leap_frog_steps(self, steps, t_step):
		self.get_V()
		self.update_momentum(t_step/2)
		for i in range(steps):
			if i%(steps/10) == 0:
				print i
				if self.gpu:
					ps.plot_phase(self.g_psi)
					ps.plot_slice(self.g_psi)
				else:
					ps.plot_phase(self.psi)
					ps.plot_slice(self.psi)
			self.update_position(t_step)
			self.get_V()
			self.update_momentum(t_step)