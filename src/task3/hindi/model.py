
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
Wrapper class using Pytorch nn.Module to create the architecture for our 
binary classification model
"""

class HindiLSTMAttentionClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_size, weights, lstm_layers, device, dropout, bidirectional):
		super(HindiLSTMAttentionClassifier, self).__init__()
		"""
        Initializes model layers and loads pre-trained embeddings from task 1
        """
		## model hyper parameters
		self.batch_size = batch_size
		self.output_size = output_size
		self.lstm_hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.lstm_layers = lstm_layers
		self.device = device
		self.bidirectional = bidirectional
		## model layers
		# initializing the look-up table.
		self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
		# assigning the look-up table to the pre-trained hindi word embeddings trained in task1.
		self.word_embeddings.weight = nn.Parameter(weights.to(self.device), requires_grad=False) 


		self.lstm = nn.LSTM(self.embedding_size, self.lstm_hidden_size, 
							num_layers=self.lstm_layers, bidirectional=self.bidirectional)
		

		self.W_s1 = nn.Linear(2*self.lstm_hidden_size, 350)
		self.W_s2 = nn.Linear(350, 30)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		
		self.fc_layer = nn.Linear(30*2*hidden_size, 2000)
		self.out = nn.Linear(2000, output_size)
		
	def init_hidden(self, batch_size):
		"""
        Initializes hidden and context weight matrix before each 
		forward pass through LSTM
        """
		layer_size = self.lstm_layers
		if self.bidirectional:
			layer_size *= 2 # since we have two layers instantiated for each lstm layer of bi-lstm
		return(Variable(torch.zeros(layer_size, batch_size, self.lstm_hidden_size).to(self.device)),
						Variable(torch.zeros(layer_size, batch_size, self.lstm_hidden_size)).to(self.device))


	def self_attention(self, lstm_out):
		attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_out)))
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

		return attn_weight_matrix
		
	def forward(self, batch, lengths):
		"""
		Performs the forward pass for each batch
        """
		##[tuple] hidden shape ((layer_size, batch_size, self.lstm_hidden_size), (layer_size, batch_size, self.lstm_hidden_size))
		self.hidden = self.init_hidden(batch.size(-1)) ## init context and hidden weights for lstm cell
		
		## batch shape: (num_sequences, batch_size)
		## embeddings shape: (seq_len, batch_size, embedding_size)
		embeddings = self.word_embeddings(batch)

		## enables the model to ignore the padded elements during backpropagation
		packed_input = pack_padded_sequence(embeddings, lengths)
		
		

		## padded_output shape : (seq_len, batch_size, num_lstm_layers * num_directions)
		output, (final_hidden_state, final_cell_state) = self.lstm(packed_input, self.hidden)
		padded_output, unpacked_lengths = pad_packed_sequence(output)
	
		#padded_out shape: (batch_size , seq_len, num_lstm_layers * num_directions)
		padded_output = padded_output.permute(1, 0, 2)
		attn_weight_matrix = self.self_attention(padded_output)

		## concat the final forward and backward hidden state
		## concat_hidden shape: (batch_size, hidden_size * num directions)
		concat_hidden = None
		if self.bidirectional:
			concat_hidden = torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim=1)	
		else:
			concat_hidden = final_hidden_state[-1]

		
		hidden_matrix = torch.bmm(attn_weight_matrix, padded_output)

		fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
		#logits = torch.relu(fc_out)
		out = self.out(fc_out)

		## passing lstm outputs to fully connected layer and then applying sigmoid activation
		## final_output shape: (batch_size, output_size)
		final_output = self.sigmoid(out) ## using sigmoid since binary labels
		
		return final_output