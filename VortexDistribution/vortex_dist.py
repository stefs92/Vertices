import numpy as np
import scipy as sp
import numpy.linalg as lin
from numba import jit, vectorize
from numba import int32
from scipy.fftpack import fftn, ifftn
from scipy.optimize import minimize
import scipy.sparse as spar
from scipy.linalg import null_space

h_bar = 1.055e-34

class vortex_dist:
	def __init__(self, res, L, rho_func, ksi, m, omega, stencil_size=3, epsilon = .1):
		self.omega = omega
		self.ksi = ksi
		self.L = L
		self.vortex = np.zeros((res, res, res, 3)) #grid of 3-vectors
		self.fft_space = np.zeros(self.vortex.shape, dtype=np.complex128)

		#if res%(stencil_size*2 + 1) != 0:
		#	res += stencil_size*2 + 1 - res%(stencil_size*2 + 1)
		self.s_dist = 2*stencil_size + 1
		self.res = res
		self.m = m
		self.mode = 'fast'
		self.epsilon = epsilon
		self.allocate_arrays()
		self.rho_func = rho_func
		self.eval_rho()
		self.stencil_maker = rbf_stencil(stencil_size)
		self.stencil_coords = self.stencil_maker.get_coord_list()
		self.differentiation_stencil()
		self.get_curl_stencil()


	def upsample(self, new_res):
		self.get_v_hat()
		new_fft_space = np.zeros((new_res, new_res, new_res, 3), dtype=np.complex128)

		slice_tup = ( slice(None, self.res//2 + 1), slice( -(self.res)//2, None) )
		for i in np.ndindex((2,2,2)):
			ind = tuple([slice_tup[j] for j in i])
			new_fft_space[ind] = self.fft_space[ind]
		if self.res%2 == 0:
			new_fft_space[self.res//2] /= 2
			new_fft_space[:, self.res//2] /= 2
			new_fft_space[:, :, self.res//2] /= 2
			new_fft_space[-self.res//2] /= 2
			new_fft_space[:, -self.res//2] /= 2
			new_fft_space[:,:,-self.res//2] /= 2

		self.fft_space = new_fft_space*(new_res**3)/(self.res**3)
		self.vortex = np.zeros(self.fft_space.shape)
		self.res = new_res
		self.allocate_arrays()
		self.get_v()
		self.get_curl_stencil()
		self.eval_rho()

	def eval_rho(self):
			# The fcc grid's axes are along the diagonal of each cube's face
			# This has a few advantages, mostly better stencils
		x = np.linspace(0, self.L, self.res + 1)[:-1]
		for i in np.ndindex((self.res, self.res, self.res)):
			print i
			self.rho[i] = self.rho_func(x[i[0]], x[i[1]], x[i[2]])


	def allocate_arrays(self):
		self.tmp_curl = np.zeros(self.vortex.shape)
		self.v_curl = np.zeros(self.vortex.shape)
		self.v_c_hat = np.zeros(self.vortex.shape, dtype=np.complex128)
		self.v_tmp = np.zeros(self.vortex.shape, dtype=np.complex128)
		self.n_v = np.zeros(self.vortex.shape)
		self.n_mag = np.zeros(self.vortex.shape[:-1])
		self.rho = np.zeros(self.vortex.shape[:-1])
		self.gradient = np.zeros(self.vortex.shape)
		self.make_energy_funcs()

	def get_v_hat(self):
		self.fft_space[...] = fftn(self.vortex, axes=(0,1,2))

	def get_v(self):
		self.vortex[...] = np.real(ifftn(self.fft_space, axes=(0, 1, 2)))

	def differentiate(self, component, axis, destination):
		slice_list = [slice(None, None) for i in range(3)]
		slice_list.append(component)
		for i in range(self.res):

			if i < self.res//2 + 1:
				slice_list[axis] = i
				destination[tuple(slice_list)] = self.fft_space[tuple(slice_list)]*i*1j*2*np.pi/self.L
			else:
				slice_list[axis] = i
				destination[tuple(slice_list)] = -self.fft_space[tuple(slice_list)]*(self.res//2 + 1 - i)*1j*2*np.pi/self.L

	def curl_spectral(self, destination=None):
		if destination is None:
			destination = self.v_curl
			self.get_v_hat()
		self.v_c_hat[...] = 0

		for i in range(3):
			self.differentiate((i + 1)%3, i, self.v_tmp)
			self.v_c_hat[..., (i - 1)%3] -= self.v_tmp[..., (i + 1)%3]
			self.differentiate(i, (i + 1)%3, self.v_tmp)
			self.v_c_hat[..., (i - 1)%3] += self.v_tmp[..., i]

		destination[...] = np.real(ifftn(self.v_c_hat, axes=(0,1,2)))

	def curl_fast(self, source = None, destination=None):
		if destination is None:
			destination = self.v_curl
		if source is None:
			source = self.vortex

		destination.flat[...] = self.curl_stencil.dot(source.reshape(np.prod(source.shape)))

	#def differentiation_stencil(self, degree):
		# Somewhat unstable way to get a stencil. Should be fine for reasonable degrees. 
	#	h = self.L/self.res
	#	if degree%2 == 1:
			#The stencils should be centered on the evaluation point
	#		degree += 1
	#	eval_mat = np.array([[(i*h)**j for i in range(-(degree/2), degree/2 + 1)] for j in range(degree + 1)])
	#	right_side = np.zeros(degree + 1)
	#	right_side[1] = 1.0
	#	self.degree = degree
	#	self.stencil = lin.solve(eval_mat, right_side)
	def differentiation_stencil(self):
		#Get stencil for the x derivative, we use symmetry to get the other stencils
		self.x_stencil = self.stencil_maker.get_deriv_stencil()



	def diff_mat_data(self, eval_component, d_component, d_axis, factor=1.0):

		#Convenience function to deal with indexing on flattened array + circulancy
		i_flat = lambda tup: np.ravel_multi_index(tup, self.vortex.shape[:-1], mode='wrap')*3
		rows = []
		cols = []
		data = []
		factor *= self.res/float(self.L)
		stencil_tmp = [i*factor for i in self.x_stencil]
		coord_tmp = np.array([[u[(k + d_axis)%3] for k in range(3)] for u in self.stencil_coords])
		for u in np.ndindex(self.vortex.shape[:-1]):
			i = i_flat(u)
			rows.extend([i + eval_component,]*(len(self.stencil_coords)))
			data.extend(stencil_tmp)
			coords = coord_tmp + u
				# the (k + d_axis)%3 is there to permute the coordinates to convert
				# a stencil for the x derivative to a stencil for some other derivative
			cols.extend([i_flat(ind) + d_component for ind in coords])
		return rows, cols, data

	def get_curl(self, source = None, destination=None):
		# Function which determines curl with a method given by 'mode'
		if self.mode == 'spectral':
			self.curl_spectral(destination = destination)
		else:
			self.curl_fast(source = source, destination = destination)

	def get_curl_stencil(self):
		rows = []
		cols = []
		data = []

		for i in range(3):
			for j in range(2):
				mat_tuple = self.diff_mat_data(i, (i + j + 1)%3, (i + 2 - j )%3, factor=(-1)**j)
				rows.extend(mat_tuple[0])
				cols.extend(mat_tuple[1])
				data.extend(mat_tuple[2])
		self.curl_stencil = spar.coo_matrix((data, (rows, cols)))
		self.curl_stencil = spar.csr_matrix(self.curl_stencil)

	def get_n_v(self):
		self.n_v[...] = 0
		self.n_v[:,:,:,2] = self.m*self.omega/(np.pi*h_bar)
		self.get_curl()
		self.n_v += self.v_curl*self.m/(np.pi*h_bar)
		self.n_mag = lin.norm(self.n_v, axis=3)

	def make_energy_funcs(self):
		c_0 = (self.m**2)/(4*np.pi*h_bar**2)
		c_1 = 2*np.pi*h_bar/self.m
		c_2 = np.pi*self.ksi**2
		c_3 = (self.L/self.res)**3
		c_4 = self.m/h_bar
		@jit(parallel=True)
		def get_energy():
			E = 0
			for i in range(self.res):
				for j in range(self.res):
					for k in range(self.res):
						E += c_0*self.rho[i,j,k]*(self.vortex[i,j,k,0]**2 + self.vortex[i,j,k,1]**2 + self.vortex[i,j,k,2]**2)
						if self.n_mag[i,j,k] < 1.0/c_2:
							E -= c_1*np.log(c_2*self.n_mag[i,j,k])*self.rho[i,j,k]*(self.n_mag[i,j,k]**2)/abs(self.v_curl[i,j,k,2] + 2*self.omega)
							E += self.rho[i,j,k]*(c_2*(self.n_mag[i,j,k]**2) - self.n_mag[i,j,k])
			return c_3*E

		@jit(parallel=True)
		def energy_gradient():
			for i in range(self.res):
				for j in range(self.res):
					for k in range(self.res):
						for l in range(3):
							self.gradient[i,j,k,l] = 2*c_0*self.rho[i,j,k]*self.vortex[i,j,k,l]
							self.tmp_curl[i,j,k,l] = -self.n_v[i,j,k,l]/abs(2*self.omega + self.v_curl[i,j,k,2])
							self.tmp_curl[i,j,k,l] *= (4*np.log(c_2*self.n_mag[i,j,k]) + 2)
							self.tmp_curl[i,j,k,l] += 2*c_2*c_4*self.n_v[i,j,k,l]/np.pi
							self.tmp_curl[i,j,k,l] -= c_4*self.n_v[i,j,k,l]/(self.n_mag[i,j,k]*np.pi)
							self.tmp_curl[i,j,k,l] *= self.rho[i,j,k]
						self.tmp_curl[i,j,k,2] += np.sign(2*self.omega + self.v_curl[i,j,k,2])*(c_1*self.rho[i,j,k])*\
												self.n_mag[i,j,k]**2/((2*self.omega + self.v_curl[i,j,k,2])**2)*np.log(c_2*self.n_mag[i,j,k])
			self.curl_gradient(self.tmp_curl,self.tmp_curl)
			for i in range(self.res):
				for j in range(self.res):
					for k in range(self.res):
						if self.n_mag[i,j,k] < 1.0/c_2:
							self.gradient[i,j,k] += self.tmp_curl[i,j,k]
			self.gradient *= c_3
			return self.gradient

		self.E_func = get_energy
		self.grad_func = energy_gradient
	
	def get_energy(self):
		self.get_n_v()
		return self.E_func()
#		E = 0
#		E += np.sum(self.m**2/(4*np.pi*h_bar**2)*self.rho*lin.norm(self.vortex, axis=3)**2)
#		 #m^2*rho*n_v^2/(4*pi*h_bar) -- kinetic term
#		a = 1/np.sqrt(np.pi*self.n_mag)
#			# upper bound of integration, used in heaviside function
#		E += np.sum((-(2*np.pi*h_bar/self.m)*np.log(np.pi*(self.ksi**2)*self.n_mag)*self.rho*(self.n_mag**2)/np.abs(self.v_curl[...,2] + 2*self.omega))[a > self.ksi])
#		# - ( log(pi*ksi^2*n_v)*rho*n_v^2/|curl(v)_z + 2*omega| ) * 2*pi*h_bar/m
#		E += np.sum((self.rho*np.pi*(self.ksi*self.n_mag)**2 - self.rho*self.n_mag)[a > self.ksi])
#		#   rho*pi*ksi^2*n_v^2 - n_v
#		return E/(self.res/self.L)**3

	def energy_gradient(self):
		self.get_n_v()
#		a = 1/np.sqrt(np.pi*self.n_mag)
#		for i in range(3):
#			self.gradient[..., i] = self.rho*self.vortex[...,i]*(self.m**2)/(2*np.pi*h_bar**2)
#		self.tmp_curl[...] = 0
#		for i in range(3):
#			self.tmp_curl[..., i] -= 4*( self.n_v[..., i]*self.rho/np.abs(2*self.omega + self.v_curl[...,2]) ) * np.log(np.pi*(self.ksi**2) * self.n_mag)
#			self.tmp_curl[..., i] -= 2*self.n_v[..., i]*self.rho/np.abs(2*self.omega + self.v_curl[...,2])
#			self.tmp_curl[..., i] += (self.ksi**2)*np.pi*self.rho*self.n_v[..., i]*self.m/h_bar - self.rho*self.n_v[..., i]*self.m/(np.pi * h_bar * self.n_mag)
#		self.curl_gradient(self.tmp_curl, self.tmp_curl)

#		self.gradient[self.ksi < a] += self.tmp_curl[self.ksi < a]
		
#		self.tmp_curl[..., :2] = 0
#		self.tmp_curl[..., 2] = np.sign(2*self.omega + self.v_curl[...,2])*(2*np.pi*h_bar*self.rho/self.m)*\
#								(self.n_mag**2/(2*self.omega + self.v_curl[...,2])**2)*np.log(np.pi * self.ksi**2 * self.n_mag)
#		self.curl_gradient(self.tmp_curl, self.tmp_curl)
#		self.gradient[self.ksi < a] += self.tmp_curl[self.ksi < a]
#		return self.gradient/(self.res/self.L)**3
		return self.grad_func()


	def curl_gradient(self, in_vec, out_vec = None):
		# Get curl of the derivative (with respect to the components of v) of an arbitrary function.
		if out_vec is None:
			out_vec = in_vec
		if self.mode == 'spectral':
			self.fft_space = fftn(in_vec, axes=(0,1,2))
			self.curl_spectral(destination=out_vec)
		else:
			self.curl_fast(source = in_vec, destination = out_vec)

	def grad_deriv(self, h=1e-5):
		# Find the derivative of the gradient with respect to a small multiple of a function
		# Part of finding the hessian

		freq_base = 2*np.pi/self.res
		hess_storage = np.empty((1 + self.degree*3, self.res, self.res, self.res, 3))

		for i in range(3):
			for j in range(self.degree):
				pass

	def integer_sin(self, axis, k_num):
		if k_num == 0:
			return lambda tup: 1.0
		else:
			freq_base = 2*np.pi/self.res
			max_freq = self.res//2
			freq = freq_base*(max_freq - k_num + 1)
			return lambda tup: np.sin(tup[axis]*freq)

	def solve(self, method = 'CG'):
 		
 		def obj_func(v):
 			self.vortex.flat[...] = v
 			E = self.get_energy()
 			print E
 			return E, self.energy_gradient().reshape(np.prod(self.gradient.shape))

 		A = minimize(obj_func, self.vortex, method=method, jac=True)
 		self.vortex.flat[...] = A.x
 		return A

 	def test_jac(self):
 		E_0 = self.get_energy()
 		G_0 = self.energy_gradient().copy()
 		for i in np.ndindex((self.res, self.res, self.res, 3)):
 			self.vortex[i] -= 1e-5
 			E = self.get_energy()
 			print i			
 			print (E_0 - E)*1e5,  G_0[i], self.vortex[i]
 			self.vortex[i] += 1e-5

 	def test_curl_deriv(self):
 		self.get_curl()
 		A = np.sum(np.exp(self.v_curl[...]))
 		B = np.empty(self.vortex.shape)
 		tmp = np.exp(self.v_curl)
 		self.curl_gradient(tmp, B)
 		for i in np.ndindex((self.res, self.res, self.res, 3)):
 			self.vortex[i] -= 1e-5
 			self.get_curl()
 			C = np.sum(np.exp(self.v_curl[...]))
 			print i			
 			print (A - C)*1e5,  B[i], self.vortex[i]
 			self.vortex[i] += 1e-5

 	def iterate_solve(self, res_list):
 		self.solve()
 		for i in res_list():
 			self.upsample(i)
 			self.solve()

# 	def get_hessian(self, h = 1e-5):
# 		hessian = np.zeros((3*self.res**3, 3*self.res**3)) #enormous array
# 		for i in np.ndindex((self.s_dist, self.s_dist, self.s_dist, 3)):


#@jitclass(spec)
class int_triple:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	def __getitem__(self, i):
		if i == 0:
			return self.x
		elif i == 1:
			return self.y
		elif i == 2:
			return self.z
		raise IndexError

	def __eq__(self, tup):
		# Equality between triples. Not intended for use with other types
		if (self[0] != tup[0]) or (self[1] != tup[1]) or (self[2] != tup[2]):
			return False
		else:
			return True

	def __add__(self, tup):
		return int_triple(self[0] + tup[0], self[1] + tup[1], self[2] + tup[2])
	
	def __sub__(self, tup):
		return int_triple(self[0] - tup[0], self[1] - tup[1], self[2] - tup[2])

	def __neg__(self):
		return int_triple(-self[0], -self[1], -self[2])

	def __abs__(self):
		# Not really abs, more like abs**2
		return (self.x**2 + self.y**2 + self.z**2)

	def __mul__(self, tup):
		val = 0.0
		for i in range(3):
			val += self[i]*tup[i]
		return val

	def __iter__(self):
		for i in range(3):
			yield self[i]

#fcc_vecs = [int_triple(1,0,0), int_triple(0,1,0), int_triple(0,0,1), int_triple(1, -1, 0), int_triple(1, 0, -1), int_triple(0, 1, -1)]
#fcc_vecs.extend([-i for i in fcc_vecs])

def mono_count(d):
	return ((d + 3)*(d + 2)*(d + 1))/6

class rbf_stencil:
	# rbf GA is too much of a pain in the ass to deal with.
	# Using polyharmonic splines
	def __init__(self, layer_num, degree = None):

		self.layer_num = layer_num
		self.get_coord_list()

		self.res = len(self.coord_list)
		self.degree = degree
		if self.degree is None:
			self.degree = [0, 2, 4, 6][min(layer_num, 3)] # might be too high
		#self.get_basis_vectors()
		self.dist_mat = np.array([[abs(i - j) for i in self.coord_list] for j in self.coord_list])
		self.x_list = np.array([i[0] for i in self.coord_list])

	def get_coord_list(self):
		self.coord_list = []
		for i in range(-self.layer_num, self.layer_num + 1):
			for j in range(-self.layer_num, self.layer_num + 1):
				for k in range(-self.layer_num, self.layer_num + 1):
					self.coord_list.append(int_triple(i,j,k))
		return self.coord_list

	def get_deriv_stencil(self):
		#Due to scale invariance, this only needs to be called once
		coord_buckets = [[],[],['']] #This is used to exhaustively list distinct monomials
		
		def eval_monomial(tup, string):
			val = 1
			for i in string:
				if i == 'x':
					val *= (tup[1] + tup[2])/np.sqrt(2)
				elif i == 'y':
					val *= (tup[0] + tup[2])/np.sqrt(2)
				elif i == 'z':
					val *= (tup[0] + tup[1])/np.sqrt(2)
			return val

		deg = self.degree
		poly_count = mono_count(deg)
		print deg, poly_count, self.res
		poly_matrix = np.zeros((poly_count, self.res))
		poly_matrix[0] = 1
		i = 1
		j = 0
		k = 0
		l = 0
		tmp_list = []
		while i < poly_count:
			if j == len(coord_buckets[k]):
				if k == 2:
					coord_buckets[l] = [x for x in tmp_list]
					tmp_list = []
					l = l+1 if l < 2 else 0
					k = l
					j = 0
				else:
					k += 1
					j = 0
			else:
#				print i,j,k,l
#				print coord_buckets[k][j] + ['x', 'y', 'z'][l]
				tmp_list.append(coord_buckets[k][j] + ['x', 'y', 'z'][l])
				poly_matrix[i] = np.array([eval_monomial(m, tmp_list[-1]) for m in self.coord_list])
				j += 1
				i += 1

		eval_mat = np.zeros((self.res + poly_count, self.res + poly_count))
		right_side = np.zeros(self.res + poly_count)

		eval_mat[:self.res, :self.res] = self.dist_mat**1.5
		right_side[:self.res] = [-3*self.x_list[i]*abs(self.coord_list[i])**.5 for i in range(self.res)]
		eval_mat[:self.res, self.res:] = poly_matrix.T
		eval_mat[self.res:, :self.res] = poly_matrix
		right_side[self.res + 1] = 1

		flat_stencil = lin.solve(eval_mat, right_side)
		print lin.cond(eval_mat)
		print np.dot(eval_mat, flat_stencil) - right_side
		return flat_stencil[:self.res]