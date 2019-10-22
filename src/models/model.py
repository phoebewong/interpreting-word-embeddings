import torch
from torch import nn
from torch.autograd import Variable


class SPINEModel(torch.nn.Module):

	def __init__(self, params):
		super(SPINEModel, self).__init__()
		
		# params
		self.inp_dim = params['inp_dim']
		self.hdim = params['hdim']
		self.noise_level = params['noise_level']
		self.getReconstructionLoss = nn.MSELoss()
		self.rho_star = 1.0 - params['sparsity']
		
		# autoencoder
		print("building model ")
		self.linear1 = nn.Linear(self.inp_dim, self.hdim) # (300, 1000) if on default
		self.linear2 = nn.Linear(self.hdim, self.inp_dim) # (1000, 300) if on default
		

	def forward(self, batch_x, batch_y):
		
		# forward
		batch_size = batch_x.data.shape[0]
		linear1_out = self.linear1(batch_x) # (batch_size, 1000) if on default
		h = linear1_out.clamp(min=0, max=1) # capped relu
		out = self.linear2(h) # (batch_size, 1000) if on default

		# sum all loss
		reconstruction_loss = self.getReconstructionLoss(out, batch_y) # reconstruction loss
		psl_loss = self._getPSLLoss(h, batch_size) 		# partial sparsity loss
		asl_loss = self._getASLLoss(h)    	# average sparsity loss
		total_loss = reconstruction_loss + psl_loss + asl_loss
		
		return out, h, total_loss, [reconstruction_loss, psl_loss, asl_loss]

	# partial sparsity loss
	def _getPSLLoss(self,h, batch_size):
		return torch.sum(h*(1-h))/ (batch_size * self.hdim) # see formula on paper

	# average sparsity loss
	def _getASLLoss(self, h):
		temp = torch.mean(h, dim=0) - self.rho_star # deviation from sparsity fraction
		temp = temp.clamp(min=0) # max of 0 and temp
		return torch.sum(temp * temp) / self.hdim # see formula on paper
