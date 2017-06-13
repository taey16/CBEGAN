from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torchvision.utils as vutils
from torch.autograd import Variable
from models import CBEGAN as net
import numpy as np
from misc import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=12, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--exp', default='interpolate', help='folder to output images')
parser.add_argument('--manualSeed', type=int, default=101, help='manual seed')
parser.add_argument('--interval', type=int, default=14, help='interval ranged from zA to zB')

opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

assert opt.netG != '', "netG must be provided!"
ngf = opt.ngf
inputChannelSize = 3
netG = net.Decoder(inputChannelSize, ngf, opt.hidden_size, True, 2)
netG.load_state_dict(torch.load(opt.netG))
print(netG)

noiseA = torch.FloatTensor(1, opt.hidden_size, 1, 1)
noiseA = Variable(noiseA)

netG.cuda()
noiseA = noiseA.cuda()

def interpolateCond(model, z, imageSize=128, intv=20):
  condition = torch.cuda.FloatTensor(intv, 2, 1, 1)

  #import pdb; pdb.set_trace()
  values = np.linspace(-1, 1, num=intv)
  condition[:,0, 0, 0].copy_(torch.from_numpy(values).cuda())
  condition[:,1, 0, 0].copy_(torch.from_numpy(-values).cuda())
  condition = Variable(condition)

  output = torch.cuda.FloatTensor(intv, 3, imageSize, imageSize).fill_(0)
  for i in range(intv):
    output[i] = model(z, condition[i].unsqueeze(0)).data.clone()
  return output


outputs = torch.FloatTensor(opt.batchSize*opt.interval, 
                            3, opt.imageSize, opt.imageSize)
for b in range(opt.batchSize):
  noiseA.data.uniform_(-1,1)
  outputs[b*opt.interval:b*opt.interval+opt.interval,:,:,:] = \
    interpolateCond(netG, noiseA, imageSize=opt.imageSize, intv=opt.interval)
vutils.save_image(outputs, 
                  '%s/interpolated_samples.png' % opt.exp, 
                  nrow=opt.interval, normalize=True)
