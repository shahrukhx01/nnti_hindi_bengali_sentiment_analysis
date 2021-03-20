import torch

"""
For centrally managing all hyper parameters, file paths and config parameters
"""

## hyper parameters for neural network

batch_size = 64
out_size = 1
hidden_size = 32
embedding_size = 300
lstm_layers = 8
epochs = 20
fc_hidden_size = 2000
is_bi_lstm = True

## self attention config
self_attention_config = {   
    'hidden_size': 200, ## refers to variable 'da' in the ICLR paper
    'output_size': 10, ## refers to variable 'r' in the ICLR paper
    'penalty': 1.0 ## refers to penalty coefficient term in the ICLR paper
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_paths = \
{
'data_file': 'data/bengali_hatespeech_subset.csv',\
'stpwds_file':'data/stopwords-bn.txt',\
'embeddings_path':'artefacts/bengali_embedding_weights_all_window_2.pickle',
'pretrained_path':'artefacts/pre_trained_hindi/hindi_classifier_attention_h{}_l{}_p{}_r{}.pth'.format(hidden_size, lstm_layers, 
                                                str(self_attention_config['penalty']).replace(".","_"), self_attention_config['output_size'])
}

## configuration dictionary
config_dict = {
    'file_paths': file_paths, 
    'batch_size': batch_size, 
    'out_size': out_size,
    'lstm_layers': lstm_layers,
    'hidden_size': hidden_size,
    'epochs': epochs,
    'embedding_size': embedding_size,
    'model_name': 'artefacts/bengali_classifier_attention_h{}_l{}_p{}_r{}'.format(hidden_size, lstm_layers, 
                                                str(self_attention_config['penalty']).replace(".","_"), self_attention_config['output_size']),
    'device': device,
    'is_bi_lstm': is_bi_lstm, 
    'self_attention_config': self_attention_config,
    'fc_hidden_size': fc_hidden_size
    }
