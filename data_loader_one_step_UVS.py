import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
#from saveNCfile import savenc



def load_test_data(FF,lead):

  psi1 = np.asarray(FF['psi1'])
  psi1 = psi1[8250:,:,:]

  psi2 = np.asarray(FF['psi2'])
  psi2 = psi2[8250:,:,:]

  m = np.asarray(FF['m'])
  m = m[8250:,:,:]  

  uv = np.zeros([np.size(psi1,0),3,128,128])


  uv[:,0,:,:] = psi1
  uv[:,1,:,:] = psi2
  uv[:,2,:,:] = m



  uv_test_input = uv[0:np.size(psi1,0)-lead,:,:,:]
  uv_test_label = uv[lead:np.size(psi1,0),:,:,:]
 


## convert to torch tensor
  uv_test_input_torch = torch.from_numpy(uv_test_input).float()
  uv_test_label_torch = torch.from_numpy(uv_test_label).float()

  return uv_test_input_torch, uv_test_label_torch


def load_train_data(GG, lead,trainN):

     psi1 = np.asarray(GG['psi1'])
     psi1 = psi1[8250:,:,:]

     psi2 = np.asarray(GG['psi2'])
     psi2 = psi2[8250:,:,:]

     m = np.asarray(GG['m'])
     m = m[8250:,:,:]

     uv = np.zeros([np.size(psi1,0),3,128,128])


     uv[:,0,:,:] = psi1
     uv[:,1,:,:] = psi2
     uv[:,2,:,:] = m




     uv_train_input = uv[0:np.size(psi1,0)-lead,:,:,:]
     uv_train_label = uv[lead:np.size(psi1,0),:,:,:]

     uv_train_input_torch = torch.from_numpy(uv_train_input).float()
     uv_train_label_torch = torch.from_numpy(uv_train_label).float()

     return uv_train_input_torch, uv_train_label_torch
