# Pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
# classic libraries
import numpy as np
#import statsmodels.api as sm    # to estimate an average with 'loess'
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('font', **font)
#rc('text', usetex=True)
import pandas as pd
import random, string
import os, time, datetime, json
# perso libraries
from utils.toolbox import *
from utils.dcgan import *
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 14}
#rc('font', **font)
#rc('text', usetex=True)
import pandas as pd
import random, string
import os, time, datetime, json
#-------------------------------------------------#
#               A) Hyper-parameters               #
#-------------------------------------------------#
CUDA = False
DATA_PATH = '../dataset/fashionmnist'
OUT_PATH = '../output'
#LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 128
IMAGE_CHANNEL = 1
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 3
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1
#-------------------------------------------------#
#               B) Data/Model/Loss                #
#-------------------------------------------------#
# B.1) dataset/corpus
#--------------------
# Corpus
# print the number of parameters
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
clear_folder(OUT_PATH)
#print("Logging to {}\n".format(LOG_FILE))
#sys.stdout = utils.StdOut(LOG_FILE)
CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))
if seed is None:
    seed = np.random.randint(1, 10000)
print("Random Seed: ", seed)
np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
cudnn.benchmark = False
device = torch.device("cuda:0" if CUDA else "cpu")

netG = Generator().to(device)
netG.apply(weights_init)
print(netG)
nbr_param = sum(p.numel() for p in netG.parameters() if p.requires_grad)
print("generator params", nbr_param)
criterion = nn.BCELoss()

netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)
nbr_param = sum(p.numel() for p in netD.parameters() if p.requires_grad)
print("discrimator params", nbr_param)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

dataset = dset.FashionMNIST(root=DATA_PATH, download=True,
                     transform=transforms.Compose([
                     transforms.Resize(X_DIM),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)
viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
counter=0
for epoch in range(EPOCH_NUM):
    for i, data in enumerate(dataloader):
        x_real = data[0].to(device)
        if counter == 0:
            print("data0.shape()", data[0].shape)
            print("data1.shape()", data[1].shape)
            print("data",data)
            print("x_real.size",x_real.shape)
            counter+=1
        real_label = torch.full((x_real.size(0),), REAL_LABEL, device=device)
        fake_label = torch.full((x_real.size(0),), FAKE_LABEL, device=device)

        # Update D with real data
        netD.zero_grad()
        y_real = netD(x_real)
        loss_D_real = criterion(y_real, real_label)
        loss_D_real.backward()

        # Update D with fake data
        z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
        x_fake = netG(z_noise)
        y_fake = netD(x_fake.detach())
        loss_D_fake = criterion(y_fake, fake_label)
        loss_D_fake.backward()
        optimizerD.step()

        # Update G with fake data
        netG.zero_grad()
        y_fake_r = netD(x_fake)
        loss_G = criterion(y_fake_r, real_label)
        loss_G.backward()
        optimizerG.step()

        if i % 100 == 0:
            print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(epoch, i, len(dataloader),loss_D_real.mean().item(),loss_D_fake.mean().item(),loss_G.mean().item()))
            vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_samples.png'), normalize=True)
            with torch.no_grad():
                viz_sample = netG(viz_noise)
                vutils.save_image(viz_sample, os.path.join(OUT_PATH, 'fake_samples_{}.png'.format(epoch)), normalize=True)
    torch.save(netG.state_dict(), os.path.join(OUT_PATH, 'netG_{}.pth'.format(epoch)))
    torch.save(netD.state_dict(), os.path.join(OUT_PATH, 'netD_{}.pth'.format(epoch)))




