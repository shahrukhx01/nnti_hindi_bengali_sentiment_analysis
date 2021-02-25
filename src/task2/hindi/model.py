
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class HindiLSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_size, weights, lstm_layers):
		super(HindiLSTMClassifier, self).__init__()
		"""
        add docs here....
        """
		
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.lstm_layers = lstm_layers
		
		# initializing the look-up table.
		self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
		# assigning the look-up table to the pre-trained hindi word embeddings trained in task1.
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) 
		self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers)
		self.out = nn.Linear(hidden_size, output_size)
		self.sigmoid = nn.Sigmoid()
		self.dropout_layer = nn.Dropout(p=0.2)
		
	def init_hidden(self, batch_size):
		"""
        add docs here....
        """
		return(Variable(torch.randn(self.lstm_layers, batch_size, self.hidden_size)),
						Variable(torch.randn(self.lstm_layers, batch_size, self.hidden_size)))


		
	def forward(self, batch, lengths):
		"""
		add docs here....
        """
		self.hidden = self.init_hidden(batch.size(-1))

		embeddings = self.word_embeddings(batch) # embedded input of shape = (batch_size, num_sequences,  embedding_size)
		#print(embeddings.shape, lengths.shape)
		packed_input = pack_padded_sequence(embeddings, lengths)
		#input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_size)
		# self.hidden = (initial_hidden_state, initial_context)
		output, (final_hidden_state, final_cell_state) = self.lstm(packed_input, self.hidden)
		#final_output = self.out(final_hidden_state[-1])
		#output = self.dropout_layer(final_hidden_state[-1])
		
		final_output = self.sigmoid(self.out(final_hidden_state[-1])) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
		
		return final_output