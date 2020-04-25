import os
import sys

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from scipy.interpolate import interp1d

from utils.toolbox import *
from utils.dcgan import *

CUDA = False     # Change to False for CPU training
VIZ_MODE = 0    # 0: random; 1: interpolation; 2: semantic calculation
OUT_PATH = '../output'
#LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 10        # Adjust this value according to your GPU memory
IMAGE_CHANNEL = 1
# IMAGE_CHANNEL = 3
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
seed = None            # Change to None to get different results at each run

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
cudnn.benchmark = False      # May train faster but cost more memory

device = torch.device("cuda:0" if CUDA else "cpu")
netG = Generator()
netG.load_state_dict(torch.load(os.path.join(OUT_PATH, 'netG_24.pth'),map_location='cpu'))
netG.to(device)

if VIZ_MODE == 0:
    viz_tensor = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
elif VIZ_MODE == 1:
    load_vector = np.loadtxt('vec_20190317-223131.txt')
    xp = [0, 1]
    yp = np.vstack([load_vector[2], load_vector[9]])   # choose two exemplar vectors
    xvals = np.linspace(0, 1, num=BATCH_SIZE)
    sample = interp1d(xp, yp, axis=0)
    viz_tensor = torch.tensor(sample(xvals).reshape(BATCH_SIZE, Z_DIM, 1, 1), dtype=torch.float32, device=device)
elif VIZ_MODE == 2:
    load_vector = np.loadtxt('vec_20190317-223131.txt')
    z1 = (load_vector[0] + load_vector[6] + load_vector[8]) / 3.
    z2 = (load_vector[1] + load_vector[2] + load_vector[4]) / 3.
    z3 = (load_vector[3] + load_vector[4] + load_vector[6]) / 3.
    z_new = z1 - z2 + z3
    sample = np.zeros(shape=(BATCH_SIZE, Z_DIM))
    for i in range(BATCH_SIZE):
        sample[i] = z_new + 0.1 * np.random.normal(-1.0, 1.0, 100)
    viz_tensor = torch.tensor(sample.reshape(BATCH_SIZE, Z_DIM, 1, 1), dtype=torch.float32, device=device)

with torch.no_grad():
    viz_sample = netG(viz_tensor)
    viz_vector = to_np(viz_tensor).reshape(BATCH_SIZE, Z_DIM)
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.savetxt('vec_{}.txt'.format(cur_time), viz_vector)
    vutils.save_image(viz_sample, 'img_{}.png'.format(cur_time), nrow=10, normalize=True)