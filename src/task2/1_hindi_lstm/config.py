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

batch_size = 32
out_size = 1
hidden_size = 32
embedding_size = 300
lstm_layers = 8
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## configuration dictionary
config_dict = {
    'file_paths': file_paths, 
    'batch_size': batch_size, 
    'out_size': out_size,
    'lstm_layers': lstm_layers,
    'hidden_size': hidden_size,
    'epochs': epochs,
    'embedding_size': embedding_size,
    'model_name': 'artefacts/hindi_classifier_h{}_l{}'.format(hidden_size, lstm_layers),
    'device': device
    }
