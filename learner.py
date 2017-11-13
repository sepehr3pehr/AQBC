import numpy as np
import scipy.io as sio
from quantizer import AQBC
import pdb

#read data ...
# map_contents = sio.loadmat('data/sift_1M.mat')
# Xtr = map_contents['learn']
# Xbase = map_contents['base']
# Xquery = map_contents['query']
# ***** initialize base query and train datase ************

learn_with_base = 0
if 'Xtr' not in locals() or 'Xbase' not in locals() or 'Xquery' not in locals():
	print("Error: Xtr or Xbase or Xquery are not initialized")
	quit()

X = Xtr
nbits = 32
epochs = 1
hasher = AQBC(X, nbits, epochs)
print("dataset = sift_1M, nbits = "  + str(nbits) + 
	" learn_with_base = " + str(learn_with_base))

hasher.optimize_all()
print "optimization done"

print "hashing dataset using learned parameters"
Bquery = hasher.hash(Xquery)
Bbase = hasher.hash(Xbase)

R = hasher.R

save_str = 'sift_1M_' + str(nbits) + '_bit_AQBC'

manifest = { # this is required for the ann benchmark
        'dataset': {
            'point_type': 'float',
            'test_size' : 1000
        }
    }
np.savez(save_str + '_base--1', data=Bbase,  R=R, manifest=[manifest])
np.savez(save_str + '_query--1', data=Bquery, R=R, manifest=[manifest])
