import torch

"""
For centrally managing all hyper parameters, file paths and config parameters
"""

## hyper parameters for neural network

batch_size = 32
num_classes = 2
hidden_size = 32
embedding_size = 300
lstm_layers = 8
epochs = 20
fc_hidden_size = 2000
is_bi_lstm = True
pretraining = False ## flag for whether to load pretrained Hindi model or not


## self attention config
self_attention_config = {   
    'hidden_size': 300, ## refers to variable 'da' in the ICLR paper
    'output_size': 10, ## refers to variable 'r' in the ICLR paper
    'penalty': 0.0 ## refers to penalty coefficient term in the ICLR paper
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bengali_file_paths = \
{
'data_file': 'data/bengali_hatespeech_subset.csv',\
'stpwds_file':'data/stopwords-bn.txt',\
'embeddings_path':'artefacts/bengali_embedding_weights_all_window_2.pickle',
'pretrained_path':'artefacts/pre_trained_hindi/hindi_classifier_attention_h{}_l{}_p{}_r{}.pth'.format(hidden_size, lstm_layers, 
                                                str(self_attention_config['penalty']).replace(".","_"), self_attention_config['output_size'])
}

hindi_file_paths = \
{
'data_file': 'data/hindi_hatespeech.tsv',\
'stpwds_file':'data/stopwords-hi.txt',\
'embeddings_path':'artefacts/embedding_weights_window_2.pickle'
}

model_name = 'sentiment_net_h{}_l{}_p{}_r{}'.format(hidden_size, lstm_layers, 
                                                str(self_attention_config['penalty']).replace(".","_"), self_attention_config['output_size'])

if pretraining:
    model_name = 'pret_' + model_name

## configuration dictionary
config_dict = {
    'bengali_file_paths': bengali_file_paths, 
    'hindi_file_paths': hindi_file_paths, 
    'batch_size': batch_size, 
    'num_classes': num_classes,
    'lstm_layers': lstm_layers,
    'hidden_size': hidden_size,
    'epochs': epochs,
    'embedding_size': embedding_size,
    'hindi_model_name': 'artefacts/hindi_{}'.format(model_name),
    'bengali_model_name': 'artefacts/bengali_{}'.format(model_name),
    'device': device,
    'is_bi_lstm': is_bi_lstm, 
    'self_attention_config': self_attention_config,
    'fc_hidden_size': fc_hidden_size,
    'pretraining': pretraining
    }
