### Has G and D networks
def get_params(l):
	params=[]
	for i in range(len(l)):
		params+=l[i].params
	return params

''' TEMPLATE
#--- Generator
class G(object):
	def __init__(self,nz):

		#- Build a net

		#- set the net in the object

		#- set the params in the object

	#- function to get the output 

		#- return the output

#--- Discriminator
class D(object):
	def __init__(self,nz):
		#- Build net

		#- set the net in the object

		#- set the params in the object

	#- function to get the output

		#- return the output
'''
def seqOutput(l,input,layer=None):
	out=l[0].get_output(input)
	if layer is None:
		for i in range(1,len(l)):
			out=l[i].get_output(out)
	#else return output for specific layer
	elif layer==0:
		return out
	else:
		for i in range(1,layer):
			out=l[i].get_output(out)
	return out

class oD4(object):
	def __init__(self,F=None):
		from theano.tensor.nnet import sigmoid, relu
		from layers import initGain, fcLayer, convUnit, nonlinLayer, reshapeLayer, convLayer
		from layers import batchNormLayer2D as bn
		l=[]
		l.append(convLayer(filterShape=(32,1,5,5),stride=2))
		l.append(bn(n_out=32))
		l.append(nonlinLayer(activation=relu))
		l.append(convLayer(filterShape=(128,32,5,5),stride=2))
		l.append(bn(n_out=128))
		l.append(nonlinLayer(activation=relu))
		l.append(convLayer(filterShape=(256,128,5,5),stride=2))
		l.append(bn(n_out=256))
		l.append(nonlinLayer(activation=relu))
		l.append(reshapeLayer((-1,256*14*14)))
		l.append(fcLayer(n_in=256*14*14, n_out=1,activation=sigmoid))
		self.l=l
		self.params=get_params(l)

	def encode(self,input):
		return seqOutput(self.l, input,layer=10)

	def get_output(self,input,layer=None):
		return seqOutput(self.l,input,layer)


class oG4(object):
	def __init__(self,nz,F=None):
		from theano.tensor.nnet import sigmoid, relu
		from layers import initGain, fcLayer, convUnit, nonlinLayer, reshapeLayer, convLayer
		from layers import batchNormLayer2D as bn
		l=[]
		l.append(fcLayer(n_in=nz,n_out=256*13*13))
		l.append(bn(n_out=256*13*13))
		l.append(nonlinLayer(activation=relu,a=0.2))
		l.append(reshapeLayer((-1,256,13,13)))
		l.append(convLayer(filterShape=(128,256,5,5),stride=0.5))
		l.append(bn(n_out=128))
		l.append(nonlinLayer(activation=relu,a=0.2))
		l.append(convLayer(filterShape=(64,128,5,5),stride=0.5))
		l.append(bn(n_out=64))
		l.append(nonlinLayer(activation=relu, a=0.2))
		l.append(convLayer(filterShape=(1,64,4,4),stride=0.5,activation=sigmoid))

		self.l=l
		self.params=get_params(l)

	def get_output(self,input,layer=None):
		return seqOutput(self.l,input,layer) #size 105



