import torch
"""
For centrally managing all hyper parameters, file paths and config parameters
"""

## hyper parameters for neural network

batch_size = 64
num_classes = 2
hidden_size = 32
embedding_size = 300
lstm_layers = 6
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_paths = \
{
'data_file': 'data/bengali_hatespeech_subset.csv',\
'stpwds_file':'data/stopwords-bn.txt',\
'embeddings_path':'artefacts/bengali_embedding_weights_all_window_2.pickle',
'pretrained_path':'artefacts/pre_trained_hindi/hindi_classifier_h{}_l{}.pth'.format(hidden_size, lstm_layers)
}

## configuration dictionary
config_dict = {
    'file_paths': file_paths, 
    'batch_size': batch_size, 
    'num_classes': num_classes,
    'lstm_layers': lstm_layers,
    'hidden_size': hidden_size,
    'epochs': epochs,
    'embedding_size': embedding_size,
    'model_name': 'artefacts/bengali_classifier_h{}_l{}'.format(hidden_size, lstm_layers),
    'device': device
    }
