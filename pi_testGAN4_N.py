from nets import oG4, oD4
from theano import tensor as T 
from piGAN import GAN
from functions import pz
import sys
sys.path.append('../../')



G=oG4
D=oD4
pz=pz('norm')
inDir='omni_back_X.npy'
exDir='Experiment/pi_oGAN4_N'

myGAN=GAN(G,D,pz,inDir,exDir)
myGAN.train()
