import torch
from torch import nn
from torch.autograd import Variable
import argparse
import utils
from utils import DataHandler
from model import SPINEModel
from random import shuffle
import numpy as np


#########################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--hdim', dest='hdim', type=int, default=1000,
                    help='resultant embedding size')

parser.add_argument('--denoising', dest='denoising',
					default=False,
					action='store_true',
                    help='noise amount for denoising auto-encoder')

parser.add_argument('--noise', dest='noise_level', type=float,
					default=0.4,
                    help='noise amount for denoising auto-encoder')

parser.add_argument('--num_epochs', dest='num_epochs', type=int,
					default=100,
                    help='number of epochs')

parser.add_argument('--batch_size', dest='batch_size', type=int,
					default=64,
                    help='batch size')

parser.add_argument('--sparsity', dest='sparsity', type=float,
					default=0.85,
                    help='sparsity')

parser.add_argument('--input', dest='input',
					required=True,
                    help='input src')

#########################################################

class Solver:

	def __init__(self, params):

		# build data handler
		self.data_handler = DataHandler()
		self.data_handler.loadData(params['input'])
		params['inp_dim'] = self.data_handler.getDataShape()[1]

		print("="*41)

		# build model
		self.model = SPINEModel(params)
		self.dtype = torch.FloatTensor

		# check if GPU is available
		use_cuda = torch.cuda.is_available()
		if use_cuda:
			self.model.cuda()
			self.dtype = torch.cuda.FloatTensor

		# set optimizer
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

		print("="*41)


	def train(self, params):
		num_epochs, batch_size = params['num_epochs'], params['batch_size'],
		optimizer = self.optimizer
		dtype = self.dtype

		# train for each epoch
		for iteration in range(num_epochs):
			self.data_handler.shuffleTrain()
			num_batches = self.data_handler.getNumberOfBatches(batch_size)
			epoch_losses = np.zeros(4) # rl, asl, psl, total

			# for each batch
			for batch_idx in range(num_batches):
				optimizer.zero_grad()
				batch_x, batch_y = self.data_handler.getBatch(batch_idx, batch_size, params['noise_level'], params['denoising'] )
				
				# transform batches into tensors
				batch_x = Variable(torch.from_numpy(batch_x), requires_grad=False).type(dtype)
				batch_y = Variable(torch.from_numpy(batch_y), requires_grad=False).type(dtype)

				# calculate losses
				out, h, loss, loss_terms = self.model(batch_x, batch_y)
				reconstruction_loss, psl_loss, asl_loss = loss_terms

				# update loss and optimizer 
				loss.backward()
				optimizer.step()

				# assign loss
				epoch_losses[0]+=reconstruction_loss.data
				epoch_losses[1]+=asl_loss.data
				epoch_losses[2]+=psl_loss.data
				epoch_losses[3]+=loss.data
			print("epoch %r: Reconstruction Loss = %.4f, ASL = %.4f, "\
						"PSL = %.4f, and total = %.4f"
						%(iteration+1, epoch_losses[0], epoch_losses[1], epoch_losses[2], epoch_losses[3]) )

	def getSpineEmbeddings(self, batch_size, params):
		ret = []
		self.data_handler.resetDataOrder()
		num_batches = self.data_handler.getNumberOfBatches(batch_size)

		# for each batch 
		for batch_idx in range(num_batches):
			batch_x, batch_y = self.data_handler.getBatch(batch_idx, batch_size, params['noise_level'], params['denoising'] )

			# transform batches into tensors
			batch_x = Variable(torch.from_numpy(batch_x), requires_grad=False).type(self.dtype)
			batch_y = Variable(torch.from_numpy(batch_y), requires_grad=False).type(self.dtype)
			_, h, _, _ = self.model(batch_x, batch_y)

			# append to embeddings
			ret.extend(h.cpu().data.numpy())
		return np.array(ret)

	def getWordsList(self):
		return self.data_handler.getWordsList()


#########################################################

def main():

	# print parameters
	params = vars(parser.parse_args())
	print("PARAMS = " + str(params))
	print("="*41)
	solver = Solver(params)
	solver.train(params)
		
	# saving the final vectors
	print("saving final SPINE embeddings")
	file_location = params['input'].split("/")[-1]
	output_path = "data/interim/" + file_location + str(params['hdim']) + "d.spine"
	final_batch_size = 512
	spine_embeddings = solver.getSpineEmbeddings(final_batch_size, params)
	utils.dump_vectors(spine_embeddings, output_path, solver.getWordsList())


if __name__ == '__main__':
	main()