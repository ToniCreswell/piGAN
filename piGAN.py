### The GAN class 

class GAN(object):
	def __init__(self,G,D,pz,inDir,exDir):

		from functions import get_args
		import theano
		self.floatX=theano.config.floatX
		
		#- get & set args
		args=get_args()
		self.args=args
		self.pzFn=pz.pz()
		#- init network and get network params
		self.G=G(self.args.nz,F=self.args.F)
		self.D=D(F=self.args.F)
		self.params_G=self.G.params
		self.params_D=self.D.params #use same params for x and G(z)
		self.prepTrain() #make all functions

		#- init error accumulators 
		self.epoch=0
		self.error_g=[]
		self.error_d=[]

		#- init directories
		self.inDir=inDir
		self.exDir=exDir
		self.oldDir=[] #used when continuing training


	#- Load data from inDir
	def loadData(self):
		#- make sure data is correct type (floatX)
		print 'loading data...'
		import cPickle as cPickle
		import numpy as np
		import theano

		try:
			f=open(self.inDir,'r')
			data=np.asarray(pickle.load(f),dtype=self.floatX)
			f.close()
		except:
			data=np.load(self.inDir).astype(self.floatX)

		data-=data.min()
		data/=data.max()
		print 'data max:', data.max(), ' min:',data.min()
		return data

	#- Write theano functions for 
	#(once compiled wont need to do again when loading models)
	def prepTrain(self):
		from theano import tensor as T
		import theano
		import numpy as np
		from theano.tensor.nnet import binary_crossentropy as Xent
		import sys
		sys.path.append('../')

		#- symbolic tensors
		x = T.tensor4('x')
		z = T.matrix('z')
		train_mode=T.iscalar('train_mode')
		#For dropout function
		self.D.train_mode=train_mode #making the train mode variable available
		self.G.train_mode=train_mode

		#Net outputs
		print 'calculating G(z), D(x) & D(G(z))'
		G_z=self.G.get_output(z)
		D_x=self.D.get_output(x)
		D_G_z=self.D.get_output(G_z)
		encoding=self.D.encode(x)

		#params
		print self.params_D, self.params_G

		#Net costs
		print 'claculating costs...'
		J_D_Dx=Xent(D_x,T.ones_like(D_x)).mean()
		J_D_DGz=Xent(D_G_z, T.zeros_like(D_G_z)).mean()
		J_G_DGz=Xent(D_G_z,T.ones_like(D_G_z)).mean()

		if self.args.pi>1 or self.args.pi<0:
			print 'pi should be betwen 0,1: using pi=0.5'
			self.args.pi=0.5
		J_D=self.args.pi*J_D_Dx + (1-self.args.pi)* J_D_DGz
		J_G=J_G_DGz

		cost=[J_D, J_G]  #not putting the others

		#Updates
		print 'calculating updates...'
		from lib.updates import Adam,Regularizer #no regulariser...
	

		updater_D=Adam(lr=self.args.lr,b1=0.5,regularizer=Regularizer(l2=1e-5))
		updater_G=Adam(lr=self.args.lr,b1=0.5,regularizer=Regularizer(l2=1e-5))

		update_D=updater_D(self.params_D,J_D)
		update_G=updater_G(self.params_G,J_G)



		#Theano functions
		print 'compiling theano functions...'
		self.train_D=theano.function([x,z],J_D,updates=update_D,givens={self.D.train_mode:np.cast['int32'](1)}, on_unused_input='ignore') #only uses givens for nets w/ dropout
		self.train_G=theano.function([z],J_G,updates=update_G,givens={self.D.train_mode:np.cast['int32'](1)}, on_unused_input='ignore')
		self.sketch=theano.function([z],G_z,givens={self.D.train_mode:np.cast['int32'](0)}, on_unused_input='ignore')
		self.encoder=theano.function([x],encoding,givens={self.D.train_mode:np.cast['int32'](0)}, on_unused_input='ignore')


	#- Training
	def train(self):
		#- imports

		print 'training...'
		from time import time
		import numpy as np
		from functions import get_batch

		x_train=self.loadData()
		print '\n  ** Data Shape:',np.shape(x_train), '  ** \n'
		print 'data loaded...'


		print 'epoch \t cost_g \t cost_d \t time'
		for i in range(self.args.maxIter):
			self.epoch+=1
			t=time()
			for j in range(self.args.k):
				Z=self.pzFn(self.args.batchSize,self.args.nz).astype(self.floatX)
				#print 'Z'
				X=get_batch(self.args.batchSize,x_train)
				#print 'X'
				cost_D=self.train_D(X,Z)
				#print 'cost_D'
			#Take another random sample...
			Z=self.pzFn(self.args.batchSize,self.args.nz).astype(self.floatX)
			cost_G=self.train_G(Z)

			print self.epoch, '\t', cost_G, '\t', cost_D, '\t', time()-t
			self.error_g.append(cost_G)
			self.error_d.append(cost_D)

		#print 'sampled from pz:', self.args.pz.name

		self.save()
	

	def visZ(self,newDir):

		import dill
		from skimage.io import imsave
		import numpy as np

		z=[]
		for i in np.arange(0.,1.,0.1):
			for j in np.arange(0.,1.,0.1):
				z.append([i,j])
		z=np.asarray(z).astype(self.floatX)
		I=self.sketch(z)

		N,sc,sx,sy=I.shape
		print N,sc,sx,sy
		montage=np.ones(shape=(10*sx,10*sy))

		n=0
		x=0
		y=0
		for i in range(10):
			for j in range(10):
				im=I[n,0,:,:]
				n+=1
				montage[x:x+sx,y:y+sy]=im
				x+=sx
			x=0
			y+=sy
		print 'montage:',np.shape(montage)
		imsave(newDir+'/montage2.png',montage)



	#- save the whole class: params and all
	def save(self):
		print 'saving...'
		import os
		from functions import gen_samples
		import sys,dill
		from matplotlib import pyplot as plt
		from skimage.io import imsave
		import numpy as np

		i=1
		while os.path.isdir(self.exDir+'/Ex_'+str(i)):
			i+=1
		newDir=self.exDir+'/Ex_'+str(i)

		self.newDir=newDir

		try:
			os.mkdir(self.newDir)
		except:
			print 'Aready Exists'

		#Text file of variables:
		f=open(newDir+'/var.txt','w')
		args=self.args
		f.write('k:'+str(args.k)+'\nlr:'+str(args.lr)+'\nnz:'+str(args.nz)+'\npi:'+str(args.pi)+'\nF:'+str(args.F)+'\nepoch:'+str(self.epoch)+'\npz:'+str(self.pzFn)+'\ninDir:'+str(self.inDir)+'\nexDir:'+str(self.exDir)+'\noldDir:'+str(self.oldDir)+'\nbatchSize'+str(args.batchSize))
		f.close()
		#Save some random generations to a sketch folder
		sketchDir=newDir+'/sketches'
		try:
			os.mkdir(sketchDir)
		except:
			print 'Already exists'


		#Save a graph of the error
		plt.plot(range(np.shape(self.error_g)[0]),self.error_g,label='G')
		plt.plot(range(np.shape(self.error_d)[0]),self.error_d,label='D')
		plt.xlabel('iter')
		plt.ylabel('cost')
		plt.legend()
		plt.savefig(self.newDir+'/cost_plot.png')

		#Save the GAN
		sys.setrecursionlimit(100000)
		dill.dump(self,open(self.newDir+'/model.pkl','wb'))


		gen_samples(self.args.batchSize,self.sketch,nz=args.nz,sketchDir=sketchDir,pz=self.pzFn)

		if self.args.nz==2:
			self.visZ(newDir)
		


		


