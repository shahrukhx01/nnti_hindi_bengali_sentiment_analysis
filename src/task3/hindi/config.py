import torch

"""
For centrally managing all hyper parameters, file paths and config parameters
"""

file_paths = \
{
'data_file': 'data/hindi_hatespeech.tsv',\
'stpwds_file':'data/stopwords-hi.txt',\
'embeddings_path':'artefacts/embedding_weights_window_2.pickle'
}

## hyper parameters for neural network

batch_size = 16
num_classes = 2
hidden_size = 32
embedding_size = 300
lstm_layers = 8
epochs = 20
fc_hidden_size = 2000
is_bi_lstm = True

## self attention config
self_attention_config = {   
    'hidden_size': 400, ## refers to variable 'da' in the ICLR paper
    'output_size': 30 ## refers to variable 'r' in the ICLR paper
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## configuration dictionary
config_dict = {
    'file_paths': file_paths, 
    'batch_size': batch_size, 
    'num_classes': num_classes,
    'lstm_layers': lstm_layers,
    'hidden_size': hidden_size,
    'epochs': epochs,
    'embedding_size': embedding_size,
    'model_name': 'artefacts/hindi_classifier_attention_h{}_l{}'.format(hidden_size, lstm_layers),
    'device': device,
    'dropout': 0.2,
    'is_bi_lstm': is_bi_lstm, 
    'self_attention_config': self_attention_config,
    'fc_hidden_size': fc_hidden_size
    }
