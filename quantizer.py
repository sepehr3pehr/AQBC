import numpy as np
import numpy.matlib as npmat
import scipy.io as sio
import pdb

class AQBC:
	def __init__(self, X, nbits, epochs):
		self.X = X
		self.n = self.X.shape[1]
		self.d = X.shape[0]
		self.nbits = nbits
		self.epochs = epochs
		self.B = np.random.randint(2, size=(self.nbits, 
			self.n))
		self.curr_obj = 0
		# R is d * c

	def objective(self):
		normB = np.linalg.norm(self.B, axis = 0)
		repNormB = npmat.repmat(normB, self.nbits, 1)
		BNormalized = np.divide(self.B, repNormB)
		RX = self.R.T.dot(self.X)
		self.obj = 0
		for i in range(self.n):
			self.obj += BNormalized[:, i].T.dot(RX[:, i])

	def optimize_B(self):
		B = np.zeros((self.nbits, self.n))
		RX = self.R.T.dot(self.X)
		for i in range(self.n):
			args_sort = np.argsort(RX[:, i].T)
			args_sort = args_sort[::-1]
			best_psi = -1 * np.inf
			for k in range(self.nbits):
				if(RX[args_sort[k], i] == 0):
					break
				b_i = np.zeros((1, self.nbits))
				b_i[:, args_sort[:k+1]] = 1
				psi = np.sum(RX[args_sort[:k+1], i]) / np.sqrt(k+1)
				if(psi > best_psi):
					best_b_i = b_i
					best_psi = psi

			self.B[:, i] = best_b_i

	def optimize_R(self):
		normB = np.linalg.norm(self.B, axis = 0)
		repNormB = np.matlib.repmat(normB, self.nbits, 1)
		BNormalized = np.divide(self.B, repNormB)
		U, _, V = np.linalg.svd(self.X.dot(BNormalized.T))
		self.R = U[:, :self.nbits].dot(V)

	def optimize_all(self):
		for i in range(self.epochs):
			print("iteration {}".format(i))
			self.optimize_R()
			self.optimize_B()
			if i%2 == 0:
				self.objective()
				print("obj @ {} is {}".format(i, self.obj))

	def hash(self, X):
		B_out = np.random.randint(2, size=(self.nbits, X.shape[1]))
		RX = self.R.T.dot(X)
		for i in range(X.shape[1]):
			args_sort = np.argsort(RX[:, i].T)
			args_sort = args_sort[::-1]
			best_psi = -1 * np.inf
			for k in range(self.nbits):
				if(RX[args_sort[k], i] == 0):
					break
				b_i = np.zeros((1, self.nbits))
				b_i[:, args_sort[:k+1]] = 1
				psi = RX[:, i].T.dot(np.squeeze(b_i)
				 / np.linalg.norm(b_i))
				if(psi > best_psi):
					best_b_i = b_i
					best_psi = psi

			B_out[:, i] = best_b_i
		return B_out

def hasher(R, nbits, X):
	B_out = np.random.randint(2, size=(nbits, X.shape[1]))
	RX = R.T.dot(X)
	for i in range(X.shape[1]):
		args_sort = np.argsort(RX[:, i].T)
		args_sort = args_sort[::-1]
		best_psi = -1 * np.inf
		for k in range(nbits):
			if(RX[args_sort[k], i] == 0):
				break
			b_i = np.zeros((1, nbits))
			b_i[:, args_sort[:k+1]] = 1
			psi = RX[:, i].T.dot(np.squeeze(b_i)
			 / np.linalg.norm(b_i))
			
			if(psi > best_psi):
				best_b_i = b_i
				best_psi = psi
		B_out[:, i] = best_b_i
	return B_out
