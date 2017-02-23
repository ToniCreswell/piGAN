Code for training a GAN on the Omniglot dataset using the network described in: 
Task Specific Adversarial Cost Function
https://arxiv.org/pdf/1609.08661.pdf

If you use this code please cite the paper:
@article{creswell2016task,
  title={Task Specific Adversarial Cost Function},
  author={Creswell, Antonia and Bharath, Anil A},
  journal={arXiv preprint arXiv:1609.08661},
  year={2016}
}

The /lib folder is taken from Newmu/dcgan_code.

To run code:
1)Install all the requirements in requirements.txt
2)Make sure the .npy.zip file is unzipped into the same location

To train a new model:

$ python pi_testGAN4.py 
or
$ python pi_testGAN4_N.py

Options include:
--nz=<size of latent>
--lr=<learning rate>
--maxIter=<number of training iterations>
--k=<no of iterations to train D>
--batchSize=<batch size>
--pi=<0.1,0.5 or 0.9> #use 0.5 for regular GAN training


To continue training a saved model:
python continue.py --exDir=<path to model.pkl file> --newIter=<number of iterations>

Models and outputs will be saved to the Experiment folder automatically. 

