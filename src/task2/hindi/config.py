file_paths = \
{
'data_file': 'data/hindi_hatespeech.tsv',\
'stpwds_file':'data/stopwords-hi.txt',\
'embeddings_path':'artefacts/embedding_weights_all_sample.pickle'
}

batch_size = 32
num_classes = 2


config_dict = {'file_paths': file_paths, 'batch_size': batch_size, 'num_classes': num_classes}
