
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
Wrapper class using Pytorch nn.Module to create the architecture for our 
binary classification model
"""

class SelfAttention(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(SelfAttention, self).__init__()
		## corresponds to variable Ws1 in ICLR paper
		self.layer1 = nn.Linear(input_size, hidden_size)
		## corresponds to variable Ws2 in ICLR paper
		self.layer2 = nn.Linear(hidden_size, output_size)

	## the forward function would receive lstm's all hidden states as input
	def forward(self, attention_input):
		## expected input shape: (batch_size , seq_len, num_lstm_layers * num_directions)
		out = self.layer1(attention_input)
		#out shape: (batch_size, seq_len, attention_hidden_size)
		out = torch.tanh(out)
		#out shape: (batch_size, seq_len, attention_out)
		out = self.layer2(out)
		## out shape post permute: (batch_size, attention_out, seq_len)
		out = out.permute(0, 2, 1)
		out = F.softmax(out, dim=2) ## softmax dimenion as per the paper

		return out ## out shape: (batch_size, attention_out, seq_len)


class HindiLSTMAttentionClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, 
				embedding_size, weights, lstm_layers, device, dropout, 
				bidirectional, self_attention_config, fc_hidden_size):
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
		self.fc_hidden_size = fc_hidden_size
		self.lstm_directions = 2  if self.bidirectional else 1 ## decide directions based on input flag

		## model layers
		# initializing the look-up table.
		self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
		# assigning the look-up table to the pre-trained hindi word embeddings trained in task1.
		self.word_embeddings.weight = nn.Parameter(weights.to(self.device), requires_grad=False) 
		
		## adding dropout to counterbalance overfitting
		self.dropout_layer = nn.Dropout(p=dropout)

		## initializng lstm layer
		self.bilstm = nn.LSTM(self.embedding_size, self.lstm_hidden_size, 
							num_layers=self.lstm_layers, bidirectional=self.bidirectional)

		## sigmoid activation for output layer as we have binary labels
		self.sigmoid = nn.Sigmoid()

		## initializing self attention layers 
		self.self_attention = None
		self.fc_layer = None
	
		## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in 
		## subsequent layers
		
		self.self_attention = SelfAttention(self.lstm_hidden_size * self.lstm_directions, 
											self_attention_config['hidden_size'], 
											self_attention_config['output_size'])
		## this layer comes right after self attention computation
		self.fc_layer = nn.Linear(self.lstm_directions * self.lstm_hidden_size * self_attention_config['output_size'], 
								  self.fc_hidden_size)
		## output layer
		self.out = nn.Linear(self.fc_hidden_size, output_size)
		
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
		
	def forward(self, batch, lengths):
		"""
		Performs the forward pass for each batch
        """
		##[tuple] hidden shape ((layer_size, batch_size, self.lstm_hidden_size), 
		# (layer_size, batch_size, self.lstm_hidden_size))
		self.hidden = self.init_hidden(batch.size(-1)) ## init context and hidden weights for lstm cell
		
		## batch shape: (num_sequences, batch_size)
		## embeddings shape: (seq_len, batch_size, embedding_size)
		embeddings = self.word_embeddings(batch)

		## enables the model to ignore the padded elements during backpropagation
		packed_input = pack_padded_sequence(embeddings, lengths)

		
		## padded_output shape : (seq_len, batch_size, num_lstm_layers * num_directions)
		output, (final_hidden_state, final_cell_state) = self.bilstm(packed_input, self.hidden)
		padded_output, unpacked_lengths = pad_packed_sequence(output)

		## here padded_output refer to matrix 'H' in the ICLR paper
		#padded_output post permute shape: (batch_size , seq_len, num_lstm_layers * num_directions)
		padded_output = padded_output.permute(1, 0, 2)

		## refers to annotation matrix 'A' in ICLR paper
		annotation_weight_matrix = self.self_attention(padded_output)	
		
		"""
		in the final step we compute output matrix 'M = AH' which has sentnece embeddings
		here the bmm (batch matrix mul) inputs have following shapes
		annotation_weight_matrix : (batch_size, attention_out, seq_len)
		padded_output: (batch_size , seq_len, lstm_hidden_size * num_directions)
		sentence_embedding shape: (batch_size , attention_out, lstm_hidden_size * num_directions)
		"""
		sentence_embedding = torch.bmm(annotation_weight_matrix, padded_output)

		"""
		transforming the two lstm directions with attention output in sentence embedding matrix 
		for fully connected layer
		sentence_embedding shape: (batch_size, lstm_directions*lstm_hidden_size*self_attention_output_size)
		"""
		sentence_embedding = sentence_embedding.view(-1, sentence_embedding.size()[1]*sentence_embedding.size()[2])

		## to counterbalance overfitting
		dropout_sentence_embedding = self.dropout_layer(sentence_embedding)
		
		## feeding dropout result to fully connected
		fc_out = self.fc_layer(dropout_sentence_embedding)
		
		## finally connecting fc output to output layer
		out = self.out(fc_out)

		## applying sigmoid activation
		## final_output shape: (batch_size, output_size)
		final_output = self.sigmoid(out) ## using sigmoid since binary labels
		
		return final_output