### Extra functions needed for training

def get_args():
	import argparse
	print '\n ** getting arguments... ** \n'
	parser=argparse.ArgumentParser()
	parser.add_argument("--nz", default=2, type=int) #length of random input
	parser.add_argument("--lr", default=0.0002, type=float) #learning rate
	parser.add_argument("--F",default=8, type=int) #width of net variable
	parser.add_argument("--maxIter",default=2000,type=int) #no of iterations
	parser.add_argument("--k",default=1,type=int) #no of iterations of training D before G
	parser.add_argument("--batchSize",default=128,type=int) #mini batch size
	parser.add_argument("--pi",default=0.5,type=float)
	return parser.parse_args()

class pz(object):
	def __init__(self,fn,a=0.,b=1.,scale=2.):
		#Return a random number generator
		pz.name='not_chosen_yet'
		self.fn=fn
		self.a=a
		self.b=b
		self.scale=scale

	def pz(self):
		if self.fn is 'norm':
			return self.normal_pz 
		if self.fn is 'gauss':
			return self.gaussian_pz
		if self.fn is 'uni':
			return self.uniform_pz
		if self.fn is 'uni_ab':
			return self.uniform_a_b_pz

	def normal_pz(self,batchSize,nz):
		import numpy as np
		self.name='normal'
		return np.asarray(np.random.normal(loc=0., scale=1., size=(batchSize,nz)))
	def gaussian_pz(self,batchSize,nz):
		import numpy as np
		self.name='gaus_sigma_'+str(self.scale)
		return np.asarray(np.random.normal(loc=0., scale=self.scale, size=(batchSize,nz)))
	def uniform_pz(self,batchSize,nz):
		import numpy as np
		self.name='uniform_0_1'
		return np.asarray(np.random.uniform(low=0., high=1., size=(batchSize,nz)))
	def uniform_a_b_pz(self,batchSize,nz):
		import numpy as np
		self.name='uniform_'+str(self.a)+'_'+str(self.b)
		return np.asarray(np.random.uniform(low=float(self.a),high=float(self.b),size=(batchSize,nz)))

def get_batch(batchSize,data):
	import numpy as np
	import theano
	floatX=theano.config.floatX
	idx=np.random.permutation(np.shape(data)[0])[0:batchSize]
	H=data.shape[2]
	out=[data[i,0,:,:] for i in idx]
	return np.asarray(out,dtype=floatX).reshape(batchSize,1,H,H)

def gen_samples(n,sketcher,nz,sketchDir,pz):
	from skimage.io import imsave
	import theano
	floatX=theano.config.floatX
	zin=pz(n,nz).astype(floatX)
	xout=sketcher(zin)
	f=open(sketchDir+'/zValues.txt','w')
	for idx in range(n):
		imsave(sketchDir+'/samples'+str(idx)+'.png', xout[idx,0,:,:])
		f.write(str(zin[idx])+'\n')


