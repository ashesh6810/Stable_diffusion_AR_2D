import numpy as np
import hdf5storage
import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import h5py
import scipy.io as sio

def load_data(filename1,filename2):
    
    f = sio.loadmat(filename1)
    omega_input = np.asarray(f['Omega'])

    f = sio.loadmat(filename2)
    omega_label = np.asarray(f['Omega'])


    return omega_input, omega_label    
