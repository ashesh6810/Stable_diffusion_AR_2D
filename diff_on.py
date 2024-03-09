import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
from data_loader_one_step_UVS import load_test_data
from data_loader_one_step_UVS import load_train_data
#from count_trainable_params import count_parameters
#import hdf5storage
from torch.optim import Adam
import math

print('Start Diff Model')

path_outputs = '/glade/derecho/scratch/asheshc/QG_moist_diffusion/outputs/'

FF=nc.Dataset('/glade/derecho/scratch/asheshc/QG_moist_diffusion/153/output.3d.nc')


Nx=128
Ny=128
Nlat =128
Nlon = 128
lead = 1
delta_t =0.01

psi_test_input_Tr_torch, psi_test_label_Tr_torch  = load_test_data(FF,lead)


M_test_level1=torch.mean((psi_test_input_Tr_torch[:,0,0:Nlat,0:Nlon].flatten()))
STD_test_level1=torch.std((psi_test_input_Tr_torch[:,0,0:Nlat,0:Nlon].flatten()))

M_test_level2=torch.mean((psi_test_input_Tr_torch[:,1,0:Nlat,0:Nlon].flatten()))
STD_test_level2=torch.std((psi_test_input_Tr_torch[:,1,0:Nlat,0:Nlon].flatten()))

M_test_level3=torch.mean((psi_test_input_Tr_torch[:,2,0:Nlat,0:Nlon].flatten()))
STD_test_level3=torch.std((psi_test_input_Tr_torch[:,2,0:Nlat,0:Nlon].flatten()))


psi_test_input_Tr_torch_norm_level1 = ((psi_test_input_Tr_torch[:,0,None,:,:]-M_test_level1)/STD_test_level1)
psi_test_label_Tr_torch_norm_level1 = ((psi_test_label_Tr_torch[:,0,None,:,:]-M_test_level1)/STD_test_level1)

psi_test_input_Tr_torch_norm_level2 = ((psi_test_input_Tr_torch[:,1,None,:,:]-M_test_level2)/STD_test_level2)
psi_test_label_Tr_torch_norm_level2 = ((psi_test_label_Tr_torch[:,1,None,:,:]-M_test_level2)/STD_test_level2)

psi_test_input_Tr_torch_norm_level3 = ((psi_test_input_Tr_torch[:,2,None,:,:]-M_test_level3)/STD_test_level3)
psi_test_label_Tr_torch_norm_level3 = ((psi_test_label_Tr_torch[:,2,None,:,:]-M_test_level3)/STD_test_level3)


psi_test_input_Tr_torch_norm = torch.cat((psi_test_input_Tr_torch_norm_level1,psi_test_input_Tr_torch_norm_level2,psi_test_input_Tr_torch_norm_level3),1)


psi_test_label_Tr_torch_norm = torch.cat((psi_test_label_Tr_torch_norm_level1,psi_test_label_Tr_torch_norm_level2,psi_test_label_Tr_torch_norm_level3),1)

print('mean value',M_test_level1)
print('std value',STD_test_level1)


print('mean value',M_test_level2)
print('std value',STD_test_level2)

print('mean value',M_test_level3)
print('std value',STD_test_level3)


def spectral_loss(output, target, wavenum_init,lamda_reg):

 loss1 = F.mse_loss(output,target) 
 # loss1 = torch.abs((output-target))/ocean_grid

 out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=3)),dim=2)
 target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=3)),dim=2)

 loss2 = torch.mean(torch.abs(out_fft[:,0,wavenum_init:]-target_fft[:,0,wavenum_init:]))
 loss3 = torch.mean(torch.abs(out_fft[:,1,wavenum_init:]-target_fft[:,1,wavenum_init:]))
 loss4 = torch.mean(torch.abs(out_fft[:,2,wavenum_init:]-target_fft[:,2,wavenum_init:]))
 
# loss = (1-lamda_reg)*loss1 + 0.33*lamda_reg*loss2 + 0.33*lamda_reg*loss2_ydir + 0.33*LC_loss
 loss = 0.25*(1-lamda_reg)*loss1 + 0.25*(lamda_reg)*loss2 + 0.25*(lamda_reg)*loss3 + 0.25*(lamda_reg)*loss4

 return loss

def RK4step(net,input_batch):
 output_1 = net(input_batch.cuda())
 output_2= net(input_batch.cuda()+0.5*output_1)
 output_3 = net(input_batch.cuda()+0.5*output_2)
 output_4 = net(input_batch.cuda()+output_3)

 return input_batch.cuda() + (output_1+2*output_2+2*output_3+output_4)/6


def Eulerstep(net,input_batch):
 output_1 = net(input_batch.cuda())
 return input_batch.cuda() + delta_t*(output_1)


def PECstep(net,input_batch):
 output_1 = net(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + delta_t*0.5*(net(input_batch.cuda())+net(output_1))


def directstep(net,input_batch):
  output_1 = net(input_batch.cuda())
  return output_1


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

def get_loss_cond(model, x_0, t, label_batch):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
   # return F.l1_loss(noise, noise_pred) + (F.mse_loss(sample_timestep(x_0,t),label_batch.cuda()))
    
    #return F.l1_loss(noise, noise_pred) + (spectral_loss(sample_timestep(x_0,t),label_batch.cuda(),wavenum_init,lamda_reg))
    return  spectral_loss((x_noisy-noise_pred),label_batch.cuda(),wavenum_init,lamda_reg)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if (t==0):
        return model_mean

    else:

        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding='same')
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 64, 64, 64, 64)
        up_channels = (64, 64, 64, 64, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding='same')

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])


        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


    
# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)



psi_test_label_Tr = psi_test_label_Tr_torch.detach().cpu().numpy()

batch_size = 100
num_epochs = 100

lamda_reg =0.2
wavenum_init=10
wavenum_init_ydir=10

model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)

fileList_train=[]


fileList_train.append('/glade/derecho/scratch/asheshc/QG_moist_diffusion/151/output.3d.nc')

for epoch in range(0, num_epochs):  # loop over the dataset multiple times

 running_loss = 0.0


 for k in fileList_train:
  print('File index',k)

  trainN=9000
  psi_train_input_Tr_torch, psi_train_label_Tr_torch  = load_train_data(nc.Dataset(k),lead,trainN)


  M_train_level1=torch.mean((psi_train_input_Tr_torch[:,0,0:Nlat,0:Nlon].flatten()))
  STD_train_level1=torch.std((psi_train_input_Tr_torch[:,0,0:Nlat,0:Nlon].flatten()))

  M_train_level2=torch.mean((psi_train_input_Tr_torch[:,1,0:Nlat,0:Nlon].flatten()))
  STD_train_level2=torch.std((psi_train_input_Tr_torch[:,1,:,:].flatten()))

  M_train_level3=torch.mean((psi_train_input_Tr_torch[:,2,0:Nlat,0:Nlon].flatten()))
  STD_train_level3=torch.std((psi_train_input_Tr_torch[:,2,0:Nlat,0:Nlon].flatten()))


  psi_train_input_Tr_torch_norm_level1 = ((psi_train_input_Tr_torch[:,0,None,:,:]-M_train_level1)/STD_train_level1)
  psi_train_label_Tr_torch_norm_level1 = ((psi_train_label_Tr_torch[:,0,None,:,:]-M_train_level1)/STD_train_level1)


  psi_train_input_Tr_torch_norm_level2 = ((psi_train_input_Tr_torch[:,1,None,:,:]-M_train_level2)/STD_train_level2)
  psi_train_label_Tr_torch_norm_level2 = ((psi_train_label_Tr_torch[:,1,None,:,:]-M_train_level2)/STD_train_level2)

  psi_train_input_Tr_torch_norm_level3 = ((psi_train_input_Tr_torch[:,2,None,:,:]-M_train_level3)/STD_train_level3)
  psi_train_label_Tr_torch_norm_level3 = ((psi_train_label_Tr_torch[:,2,None,:,:]-M_train_level3)/STD_train_level3)

  psi_train_input_Tr_torch_norm = torch.cat((psi_train_input_Tr_torch_norm_level1,psi_train_input_Tr_torch_norm_level2,psi_train_input_Tr_torch_norm_level3),1)


  psi_train_label_Tr_torch_norm = torch.cat((psi_train_label_Tr_torch_norm_level1,psi_train_label_Tr_torch_norm_level2,psi_train_label_Tr_torch_norm_level3),1)




  for step in range(0,trainN-batch_size,batch_size):
        # get the inputs; data is a list of [inputs, labels]
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
        input_batch, label_batch = psi_train_input_Tr_torch_norm[indices,:,:,:], psi_train_label_Tr_torch_norm[indices,:,:,:]


        

        print('shape of input', input_batch.shape)
        print('shape of label1', label_batch.shape)
        # zero the parameter gradients
        optimizer.zero_grad()
        t = torch.randint(0, T, (batch_size,), device=device).long()
        loss = get_loss_cond(model, input_batch.float().cuda(), t, label_batch.float().cuda())
        loss.backward()
        optimizer.step()
        print('Epoch',epoch)
        print('Step',step)
        print('Loss',loss)

torch.save(model.state_dict(), './Diffusion_FFT_spectralloss_lead'+str(lead)+'.pt')
print('Model saved')


psi_test_label_Tr_torch_denorm = psi_test_label_Tr_torch_norm_level1*STD_test_level1+M_test_level1
psi_test_label_Tr = psi_test_label_Tr_torch.detach().cpu().numpy()
Nens = 20
Nsteps = 500
pred = np.zeros([Nsteps,Nens,3,Nx,Nx])
#tt =  torch.randint(0, T, (1,), device=device).long()
#print(np.shape(np.squeeze((sample_timestep (input_data_torch_norm_level1[0,:,:,:].reshape([1,1,Nx,Nx]),tt).detach().cpu().numpy()))))

for k in range(0,Nsteps):

 print('time step',k)   
 if (k==0):

   for ens in range (0,Nens):

    tt =  torch.randint(0, T, (1,), device=device).long()
    x_noisy, noise = forward_diffusion_sample(psi_test_input_Tr_torch_norm[0,:,:,:].reshape([1,3,Nx,Ny]).float().cuda(), tt, device)
#    u=psi_test_input_Tr_torch_norm[0,:,:,:].reshape([1,3,Nx,Ny]).float().cuda()+x_noisy - model(x_noisy, tt)
    u=x_noisy - model(x_noisy, tt)

#    pred[k,ens,:,:,:] = np.squeeze(sample_timestep (model,psi_test_input_Tr_torch_norm[0,:,:,:].reshape([1,3,Nx,Ny]).float().cuda(),tt).detach().cpu().numpy())
    pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
  
 else:

   mean_traj = torch.from_numpy(np.mean(pred [k-1,:,:,:,:],0).reshape([1,3,Nx,Ny])).float().cuda()  
   for ens in range (0, Nens):
     tt =  torch.randint(0, T, (1,), device=device).long()
     x_noisy, noise = forward_diffusion_sample(mean_traj, tt, device)
     u=x_noisy - model(x_noisy, tt)
    
#    pred[k,ens,:,:,:] = np.squeeze(sample_timestep (model,psi_test_input_Tr_torch_norm[0,:,:,:].reshape([1,3,Nx,Ny]).float().cuda(),tt).detach().cpu().numpy())
     pred[k,ens,:,:,:] = np.squeeze(u.detach().cpu().numpy())
     
     #pred[k,ens,:,:,:] = np.squeeze(model,sample_timestep(model,mean_traj,tt).detach().cpu().numpy())
 

STD_test_level1=STD_test_level1.detach().cpu().numpy()
M_test_level1=M_test_level1.detach().cpu().numpy()

STD_test_level2=STD_test_level2.detach().cpu().numpy()
M_test_level2=M_test_level2.detach().cpu().numpy()

STD_test_level3=STD_test_level3.detach().cpu().numpy()
M_test_level3=M_test_level3.detach().cpu().numpy()


pred_denorm1 = pred [:,:,0,None,:,:]*STD_test_level1+M_test_level1
pred_denorm2 = pred [:,:,1,None,:,:]*STD_test_level2+M_test_level2
pred_denorm3 = pred [:,:,2,None,:,:]*STD_test_level3+M_test_level3

pred = np.concatenate((pred_denorm1,pred_denorm2,pred_denorm3),axis=2)

np.savez(path_outputs+'predicted_QG_spectral_loss_diffusion_lamda_'+str(lamda),pred,psi_test_label_Tr)

'''
matfiledata = {}
matfiledata[u'prediction'] = pred
matfiledata[u'Truth'] = psi_test_label_Tr
hdf5storage.write(matfiledata, '.', path_outputs+'predicted_QG_spectral_loss_diffusion'+'.mat', matlab_compatible=True)
'''
print('Saved Predictions')
