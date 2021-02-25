import pandas as pd
from preprocess import Preprocess
import logging
import torch
from torch.utils.data import DataLoader
from hasoc_dataset import HASOCDataset
logging.basicConfig(level=logging.INFO)

"""
For loading HASOC data and preprocessing
"""
class HASOCData:
    def __init__(self, file_path):

        self.load_data(file_path)
        self.create_vocab()
        
    def load_data(self, file_paths):
        logging.info('loading and preprocessing data...')
        self.data = pd.read_csv(file_paths['data_file'], sep='\t') ## reading data file
        self.data = Preprocess(file_paths['stpwds_file']).perform_preprocessing(self.data) ## performing text preprocessing
        logging.info('reading and preprocessing data completed...')
    
    def create_vocab(self):
        logging.info('creating vocabulary...')
        self.vocab = list(self.data.clean_text.str.split(expand=True).stack().value_counts().keys())
        self.word2index = {word:index for index,word in enumerate(self.vocab)}
        self.index2word = {index:word for index,word in enumerate(self.vocab)}
        logging.info('creating vocabulary completed...')

    def transform_labels(self):
        self.data['labels'] = self.data.task_1.map({'HOF': 1, 'NOT': 0})

    def data2tensors(self):
        self.transform_labels()
        vectorized_sequences, sequence_lengths, targets = [], [], []
        raw_data = list(self.data.clean_text.values)

        ## get the text sequence from dataframe
        for index, sentence in enumerate(raw_data):
            ## convert sentence into vectorized form replacing words with vocab indices
            vectorized_sequence = self.vectorize_sequence(sentence)
            sequence_length = len(vectorized_sequence) ## computing sequence lengths for padding
            if sequence_length <= 0:
                continue
            
            vectorized_sequences.append(vectorized_sequence) ## adding sequence vectors to train matrix  
            sequence_lengths.append(sequence_length) 
            targets.append(self.data.labels.values[index]) ## fetching label for this example
        
         
        ## padding zeros at the end of tensor till max length tensor
        padded_sequence_tensor = self.pad_sequences(vectorized_sequences, torch.LongTensor(sequence_lengths))
        length_tensor = torch.LongTensor(sequence_lengths) ## casting to long 
        target_tensor = torch.LongTensor(targets) ## casting to long 

        return (padded_sequence_tensor, target_tensor, length_tensor, raw_data )


    def get_data_loader(self, batch_size = 8):
        padded_sequence_tensor, target_tensor, length_tensor, raw_data = self.data2tensors()
        hasoc_dataset = HASOCDataset(padded_sequence_tensor, target_tensor, length_tensor, raw_data)
        hasoc_dataloader = DataLoader(hasoc_dataset, batch_size=batch_size)

        return hasoc_dataloader

    def sort_batch(self, batch, targets, lengths):
        sequence_lengths, perm_idx = lengths.sort(0, descending=True)
        sequence_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return sequence_tensor.transpose(0, 1), target_tensor, sequence_lengths

    def vectorize_sequence(self, sentence):
        return [self.word2index[token] for token in sentence.split()]
    
    def pad_sequences(self, vectorized_sequences, sequence_lengths):
        padded_sequence_tensor = torch.zeros((len(vectorized_sequences), sequence_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(vectorized_sequences, sequence_lengths)):
            padded_sequence_tensor[idx, :seqlen] = torch.LongTensor(seq)
        return padded_sequence_tensor
    