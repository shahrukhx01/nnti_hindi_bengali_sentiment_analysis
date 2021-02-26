file_paths = \
{
'data_file': 'data/hindi_hatespeech.tsv',\
'stpwds_file':'data/stopwords-hi.txt',\
'embeddings_path':'artefacts/embedding_weights_all_sample.pickle'
}

## hyper params

batch_size = 32
num_classes = 2
hidden_size = 32
lstm_layers = 1
epochs = 25

## configuration dictionary
config_dict = {
    'file_paths': file_paths, 
    'batch_size': batch_size, 
    'num_classes': num_classes,
    'lstm_layers': lstm_layers,
    'hidden_size': hidden_size,
    'epochs': epochs,
    'model_name': 'hindi_classifier_h{}_l{}'.format(hidden_size, lstm_layers)
    }
