
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
Wrapper class using Pytorch nn.Module to create the architecture for our 
binary classification model
"""
class BengaliLSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_size, weights, 
												lstm_layers, device, pretrained_state_dict_path):
		super(BengaliLSTMClassifier, self).__init__()
		"""
        Initializes model layers and loads pre-trained embeddings from task 1
        """
		## model hyper parameters
		self.pretrained = pretrained_state_dict_path
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.lstm_layers = lstm_layers
		self.device = device
		
		## model layers
		# initializing the look-up table.
		self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
		# assigning the look-up table to the pre-trained bengali word embeddings trained in prior to this.
		self.word_embeddings.weight = nn.Parameter(weights.to(self.device), requires_grad=False) 


		self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers)
		self.out = nn.Linear(hidden_size, output_size)
		self.sigmoid = nn.Sigmoid()
		self.dropout_layer = nn.Dropout(p=0.2)
		
	def init_hidden(self, batch_size):
		"""
        Initializes hidden and context weight matrix before each 
		forward pass through LSTM
        """
		return(Variable(torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(self.device)),
						Variable(torch.zeros(self.lstm_layers, batch_size, self.hidden_size)).to(self.device))
		
	def load_pretrained_layers(self):
		"""
        Loads pretrained LSTM and FC layers from hindi classifier
        """
		try:
			state_dict = torch.load(self.pretrained) ## load pretrained weights
		except:
			print("No pretrained model exists for current architecture!")

		print('Loading pretrained weights...')

		with torch.no_grad():
			self.lstm.weight.copy_(state_dict['lstm.weight'])
			self.lstm.bias.copy_(state_dict['lstm.bias'])

			self.out.weight.copy_(state_dict['out.weight'])
			self.out.bias.copy_(state_dict['out.bias'])



		
	def forward(self, batch, lengths):
		"""
		Performs the forward pass for each batch
        """
		self.hidden = self.init_hidden(batch.size(-1)) ## init context and hidden weights for lstm cell

		embeddings = self.word_embeddings(batch) # embedded input of shape = (batch_size, num_sequences,  embedding_size)
		packed_input = pack_padded_sequence(embeddings, lengths)
		output, (final_hidden_state, final_cell_state) = self.lstm(packed_input, self.hidden)
		output = self.dropout_layer(final_hidden_state[-1]) ## to avoid overfitting
		final_output = self.sigmoid(self.out(output)) ## using sigmoid since binary labels
		
		return final_output