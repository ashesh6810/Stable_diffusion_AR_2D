import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from data_loader_one_step_UVS import load_test_data
from data_loader_one_step_UVS import load_train_data


print('Start VAE')
lead = 1
Nx=128
Ny=128
Nlat=128
Nlon=128

path_outputs = '/glade/derecho/scratch/asheshc/QG_moist_diffusion/outputs/'

FF=nc.Dataset('/glade/derecho/scratch/asheshc/QG_moist_diffusion/153/output.3d.nc')

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dimx=128
dimy=128
num_filters=64

class VAE(nn.Module):
    def __init__(self, imgChannels=3, out_channels=3, featureDim=num_filters*dimx*dimy, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = (nn.Conv2d(imgChannels, num_filters, kernel_size=5, stride=1, padding='same'))
        self.encConv2 = (nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding='same'))
        self.encConv3 = (nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding='same'))
        self.encConv4 = (nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding='same'))
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = (nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding='same' ))
        self.decConv2 = (nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding='same' ))
        self.decConv3 = (nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding='same' ))
        self.decConv4 = (nn.Conv2d(num_filters, out_channels, kernel_size=5, stride=1, padding='same' ))
        

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = F.relu(self.encConv4(x))
        x = x.view(-1, num_filters*dimx*dimy)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, num_filters, dimx, dimy)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = F.relu(self.decConv3(x))
        x = (self.decConv4(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar


learning_rate = 1e-4
beta=1.0
batch_size = 10
num_epochs = 100
lamda_reg =0.0
wavenum_init=10
wavenum_init_ydir=10

net = VAE()
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

fileList=[151,155,156,157,159]
fileList_train=[]

for k in fileList:

    fileList_train.append('/glade/derecho/scratch/asheshc/QG_moist_diffusion/'+str(k)+'/output.3d.nc')
    
#fileList_train.append('/glade/derecho/scratch/asheshc/QG_moist_diffusion/151/output.3d.nc')


for epoch in range(0, num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for k in fileList_train:
     print('Training file index',k)
     FF=nc.Dataset(k) 
     trainN=9000
     psi_train_input_Tr_torch, psi_train_label_Tr_torch  = load_train_data(FF,lead,trainN)
     



###### normalize each batch ##########
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

     for step in range(0,trainN,batch_size):
        # get the inputs; data is a list of [inputs, labels]
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
        input_batch, label_batch = psi_train_input_Tr_torch[indices,:,:,:], psi_train_label_Tr_torch[indices,:,:,:]
        print('shape of input', input_batch.shape)
        print('shape of output', label_batch.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
#        output,_,_,_,_,_,_ = net(input_batch.cuda())
        output, mu, logVar = net(input_batch.cuda())
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
#        loss = spectral_loss(output,label_batch.cuda(),wavenum_init,lamda_reg) + beta*kl_divergence 
        loss = F.mse_loss(output,label_batch.cuda())+beta*kl_divergence
        loss.backward()
        optimizer.step()
        print('Loss',loss)
#        if step % 100 == 0:    # print every 2000 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, step + 1, loss))

print('Finished Training')


torch.save(net.state_dict(), './BNN_VAE_lambda'+str(lamda_reg)+'.pt')

print('BNN Model Saved')


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
    output,_,_ = net(psi_test_input_Tr_torch_norm[0,:,:,:].reshape([1,3,Nx,Ny]).float().cuda())
    pred[k,ens,:,:,:] = np.squeeze(output.detach().cpu().numpy())

 else:

   mean_traj = torch.from_numpy(np.mean(pred [k-1,:,:,:,:],0).reshape([1,3,Nx,Ny])).float().cuda()
   for ens in range (0, Nens):
     output,_,_ = net(mean_traj) 
     pred[k,ens,:,:,:] = np.squeeze(output.detach().cpu().numpy())

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

np.savez(path_outputs+'predicted_QG_spectral_loss_VAE_lamda_'+str(lamda),pred,psi_test_label_Tr)

