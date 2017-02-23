### Layers
from matplotlib import pyplot as plt
import cPickle as pickle 
from theano import tensor as T
from skimage.io import imsave
import numpy as np
import theano
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import relu, sigmoid
from theano.tensor.nnet.abstract_conv import bilinear_upsampling
from numpy.random import binomial 
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
rand=RandomStreams()
floatX=theano.config.floatX


#--- Weight Initialisers
#- inintW (uniform fan_in/fan_out)
def initW(filterShape):
	fan_in=np.prod([filterShape[1:]])
	fan_out=np.prod([filterShape[2:]])*filterShape[0]
	bound=np.sqrt(6./(fan_in+fan_out))
	Winit=np.random.uniform(low=-bound,high=bound,size=filterShape)
	Winit=theano.shared(np.asarray(Winit, dtype=floatX),borrow=True,name='W')
	return Winit	

#- initBias (all zeros)
def initBias(size):
	b=np.zeros(size)
	return theano.shared(np.asarray(b,dtype=floatX),borrow=True,name='b')

#- initGain (all ones)  
	ginit=np.ones(size)
	return theano.shared(np.asarray(ginit,dtype=floatX),borrow=True,name='g')
	

#- fcWinit (based on n_in/n_out)
def fcWinit(n_in,n_out):
	bound=np.sqrt(6./(n_in+n_out))
	Winit=np.random.uniform(low=-bound, high=bound, size=(n_in,n_out))
	return theano.shared(np.asarray(Winit, dtype=floatX),borrow=True,name='fcW')

#--- convLayer (with stride)
class convLayer(object):
	def __init__(self,filterShape,stride,activation=None):

		#input: [N,channels,width,height]
		#filterShape: [out_channels,in_channels,h,w] 
			#filter size should be odd
		#stride: interger or 1./interger
		#activation: nnet.relu, nnet.sigmoid etc....
		
		self.W=initW(filterShape)
		self.b=initBias(filterShape[0])

		self.params=[self.W,self.b]
		self.activation=activation
		self.stride=stride

	def get_output(self,input):
		#apply convolution
		if self.stride >= 1: 
			conv=conv2d(input=input,filters=self.W,subsample=(self.stride,self.stride), border_mode='half') #same size
		else:
			#Not proper deconv: conv + upsampling
			try: 
				ratio=int(1./self.stride) 
			except:
				print '\n \t why is ratio 0? \n \t WARNING \n \t assuming stride=0.5 \n'
				ratio=2
			up=bilinear_upsampling(input,ratio)
			conv=conv2d(input=up,filters=self.W,subsample=(1,1), border_mode='half')


		#apply activation
		if self.activation is None:
			self.output=conv+self.b.dimshuffle('x', 0, 'x', 'x')
		else:
			self.output=self.activation(conv+self.b.dimshuffle('x', 0, 'x', 'x'))
		return self.output



#--- batchNormLayer(2d)
class batchNormLayer2D(object):
	def __init__(self,n_out):

		#Need to be the same shape as the input (which is the output from the layer before)
		self.gain = initGain((n_out))
		self.bias = initBias((n_out))
		self.mean = initBias((n_out))
		self.std = initGain((n_out))

		#Trainable parameters
		self.params=[self.gain,self.bias]#,self.mean,self.std]

	def get_output(self,input):

		from lib.ops import batchnorm as bn
		self.output=bn(input, self.gain,self.bias)
		return self.output
		



#--- fcLayer
class fcLayer(object):
	def __init__ (self,n_in,n_out,activation=None,a=None):

		self.W=fcWinit(n_in,n_out)
		self.b=initBias(n_out)
		self.a=a

		self.params=[self.W,self.b]
		self.activation=activation

	def get_output(self,input):
		self.output=(T.dot(input,self.W)+self.b)
		if self.activation is None: #Allows bn to be applied before activation
			return self.output
		elif self.a is None:
			self.output=self.activation(self.output)
			return self.output
		else:
			self.output=self.activation(self.output,self.a)
			return self.output

#--- Units (deconv and conv in one)
class convUnit(object):
	def __init__(self,filterShape,stride,activation,a=None):
		# a can only be scalar fixed atm
		#- a -- theano shared variable e.g. for a leaky relu  ##NOT LEARNABLE
		self.conv=convLayer(filterShape=filterShape,stride=stride)
		self.batch=batchNormLayer2D(n_out=filterShape[0])
		self.params=self.conv.params+self.batch.params  #to make learnable add here
		self.activation=activation
		self.a=a
	def get_output(self,input):
		if self.a is None:
			return self.activation(self.batch.get_output(self.conv.get_output(input)))
		else: #For leaky relu or other activation functions with learnable parameters
			return self.activation(self.batch.get_output(self.conv.get_output(input)),self.a)

class nonlinLayer(object):
	def __init__(self, activation,a=None):
		self.activation=activation
		self.a=a
		self.params=[]
	def get_output(self,input):
		if self.a is None:
			return self.activation(input)
		else:
			return self.activation(input,self.a) 

class reshapeLayer(object):
	def __init__(self,newShape):
		self.newShape=newShape
		self.params=[]
	def get_output(self,input):
		try:
			return input.reshape(self.newShape)
		except AttributeError:
			return np.reshape(input,(self.newShape))

#--- dropLayer (needs switch which will be a symbolic var)(just a fn - no params)
def dropLayer(input,p=0.5,switch=1):
	#switch is a theano.iscalar
	import theano
	from theano import tensor as T
	floatX=theano.config.floatX
	rng=T.shared_randomstreams.RandomStreams(np.random.randint(low=1,high=1234))
	mask=rng.binomial(n=1, p=p, size=(input.shape),dtype=floatX)
	return T.switch(T.neq(switch,0),input*mask/p, input)
