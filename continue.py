### Continue training a model

#get args
def get_args():
	import argparse
	p=argparse.ArgumentParser()
	p.add_argument("--exDir", required=True, type=str)
	p.add_argument("--newIter",required=True, type=int)
	return p.parse_args()

#load model
def cont():
	import dill
	args=get_args()
	f=open(args.exDir+'/model.pkl')
	model=dill.load(f)
	f.close()
	try:			
		model.args.maxIter=args.newIter #newr version of GAN
	except:
		model.maxIter=args.newIter
	model.train()
	model.oldDir=args.exDir

#contine training...
cont()