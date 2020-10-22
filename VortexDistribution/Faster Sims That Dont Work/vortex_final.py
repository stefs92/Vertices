import numpy as np
import scipy as sp
import numpy.linalg as lin
import scipy.sparse as spar
from scipy.optimize import minimize
from numba import jit

h_bar = 1.0
sec_stencil = np.array([-.5, 0, .5])
tri_stencil = np.array([1.0/12, -8.0/12, 0, 8.0/12, -1.0/12])



@jit
def curl_mat_r(L, res, stencil=sec_stencil):
	#BC's = odd about x = 0, zero for |y| > -L/2, x > L 
	# This method constructs a discrete operator for the rho component of the curl
	h = float(L)/float(res)
	stencil_size = stencil.shape[0] - 1 # Assumed to be a centered stencil
	half_size = stencil_size//2
	mat_length = (stencil_size)*res**2 - res*(half_size**2 + half_size)       # subtraction accounts for stencil entries that would
																		      # lie outside the domain.
	row = np.empty(mat_length, dtype=int)
	col = row.copy()
	data = np.empty(mat_length)
	ind = 0
	A = np.arange(stencil_size + 1)
	stencil_indices = A[A != half_size] - half_size
	for i in range(res):
		for j in range(res):
			for k in stencil_indices:
				if (j + k < res) and (j + k >= 0):
					row[ind] = i*res + j
					col[ind] = i*res + j + k
					data[ind] = -stencil[k + half_size]/h
					ind += 1

	return spar.csr_matrix((data, (row, col)))


@jit
def curl_mat_z(L, res, stencil=sec_stencil):
	#BC's = odd about x = 0, zero for |y| > -L/2, x > L; This constructs a discrete operator for the z component of the vector curl 
	h = float(L)/float(res)
	stencil_size = stencil.shape[0] - 1 # Assumed to be a centered stencil
	half_size = stencil_size//2
	mat_length = (stencil_size)*res**2 - res*(half_size**2 + half_size)//2 # subtraction accounts for stencil entries that would
																		      # lie outside the domain.
	row = np.empty(mat_length, dtype=int)
	col = row.copy()
	data = np.empty(mat_length)
	ind = 0
	A = np.arange(stencil_size + 1)
	stencil_indices = A[A != half_size] - half_size

	for i in range(res):
		for j in range(res):
			# Construct arrays of row, col, and data for constructing a sparse matrix
			# There can be repeated row, col pairs, but this is handled by 'spar.csr_matrix'
			#row[ind] = i*res + j
			#col[ind] = i*res + j
			#data[ind] = 1.0/(h*(i + .5))
			#ind += 1
			for k in stencil_indices:
				if (i + k < res):
					row[ind] = i*res + j
					col[ind] = (i + k)*res + j if i + k >= 0 else -(i + k + 1)*res + j
					data[ind] = ((h*(i + k + .5))/(h*(i + .5)))*stencil[k + half_size]/h
					if (i + k < 0): 
						data[ind] *= -1
					ind += 1

	return spar.csr_matrix((data, (row, col)))

class vortex_dist:
	def __init__(self, rho_func, res = 6, W = 1.0, L = 1.0, stencil = sec_stencil):
		self.W = W
		self.L = L
		self.res = res
		self.h = self.L/float(self.res)
		self.allocate_arrays()
		self.rho_func = rho_func
		self.stencil = stencil
		self.stencil_size = (len(stencil) - 1)/2
		self.eval_rho()
		self.v[...] = 0
		self.make_curl_stencil()

	def allocate_arrays(self):
		self.v = np.empty((self.res, self.res), dtype=np.float64)
		self.rho = np.empty((self.res, self.res))                # Density multiplied by radius
		self.curl = np.empty((2, self.res, self.res), dtype = np.float64)
		self.n_mag = np.empty((self.res, self.res), dtype=np.float64)
		self.n_v = np.empty((2, self.res, self.res), dtype=np.float64)
		self.E_tmp = np.empty((self.res, self.res), dtype=np.float64)
		self.grad = self.E_tmp.copy()   
		self.grad_tmp = np.empty((2, self.res, self.res), dtype=np.float64)

	def eval_rho(self):
		x = np.linspace(0, self.L, self.res + 1)[:-1]
		x += x[1]/2
		y = x.copy() - .5*self.L
		Y, X = np.meshgrid(y, x) # meshgrid uses the opposite convention I do for coordinates.
		self.rho = self.rho_func(X, Y)*X
		self.center_rho = np.array([self.rho_func(i, 0.0) for i in y])

	def make_curl_stencil(self):
		self.z_op = curl_mat_z(self.L, self.res, self.stencil)
		self.r_op = curl_mat_r(self.L, self.res, self.stencil)
		self.z_t = self.z_op.transpose(copy=True)
		self.r_t = self.r_op.transpose(copy=True)

	def get_curl(self):
		self.curl[0].flat[...] = self.r_op.dot(self.v.flat)
		self.curl[1].flat[...] = self.z_op.dot(self.v.flat)
		self.n_v[...] = self.curl*.5
		self.n_v[1] += self.W
		self.n_mag[...] = lin.norm(self.n_v, axis=0)

	def get_energy(self):
		# Evaluate energy functional
		self.get_curl()
		self.E_tmp[...] = np.log(self.n_mag)
		self.E_tmp *= self.n_mag**2
		#self.E_tmp /= np.abs(self.n_v[1]) + 1e-5
		self.E_tmp += .25*self.v**2
		self.E_tmp += self.n_mag**2
		self.E_tmp -= self.n_mag
		self.E_tmp *= self.rho

		#Singular component:
		self.E_tmp[0] += .5*(self.v[0]**2)*self.center_rho

		return np.sum(self.E_tmp)/self.res**2
		# No penalty atm

	def get_gradient(self, NEED_CURL=True):
		# Gradient of the energy functional, uses the anti-symmetric property of the curl operator.
		if NEED_CURL:
			self.get_curl()
		self.grad_tmp[0] = self.n_v[0]*np.log(self.n_mag)#/(np.abs(self.n_v[1]) + 1e-5)
		self.grad_tmp[1] = self.n_v[1]*np.log(self.n_mag)#/(np.abs(self.n_v[1]) + 1e-5)
		self.grad_tmp[0] += self.n_v[0]#/(np.abs(2*self.n_v[1]) + 1e-5)
		self.grad_tmp[1] += self.n_v[1]#/(np.abs(2*self.n_v[1]) + 1e-5)
		#self.grad_tmp[1] -= .5*np.sign(self.n_v[1])*(self.n_mag**2)*np.log(self.n_mag)/(np.abs(self.n_v[1]) + 1e-5)**2
		self.grad_tmp[...] += self.n_v
		self.grad_tmp[0] -= .5*self.n_v[0]/self.n_mag
		self.grad_tmp[1] -= .5*self.n_v[1]/self.n_mag
		self.grad_tmp[0] *= self.rho
		self.grad_tmp[1] *= self.rho
		self.grad.flat[...] = self.r_t.dot(self.grad_tmp[0].flat[...])
		self.grad.flat[...] += self.z_t.dot(self.grad_tmp[1].flat[...])
		self.grad.flat[...] += .5*self.v.flat[...]*self.rho.flat[...]
		self.grad.flat[...] *= 1.0/self.res**2

		#Singular component
		self.grad[0] += self.v[0]*self.center_rho

		return self.grad.flat[...]

	def solve(self):
		def obj_func(v):
			self.v.flat[...] = v
			E = self.get_energy()
			grad = self.get_gradient()
			return E, grad.flat[...]

		A = minimize(obj_func, self.v.flat[...], method='CG', jac=True)
		self.v.flat[...] = A.x[...]
		return A

	def grad_test(self, h = .0001):
		grad = self.get_gradient().flat[...].copy()
		for i in range(self.res**2):
			self.v.flat[i] += h/2
			E_hi = self.get_energy()
			self.v.flat[i] -= h
			E_lo = self.get_energy()
			print (E_hi - E_lo)/h, grad[i]
			self.v.flat[i] += h/2
