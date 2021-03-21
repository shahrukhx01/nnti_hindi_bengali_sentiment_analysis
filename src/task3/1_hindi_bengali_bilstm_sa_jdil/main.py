from bengali_data import BengaliData
from hindi_data import HASOCData
from model import SentimentNet
from train import train_model 
from hindi_eval import evaluate_hindi_test_set
from bengali_eval import evaluate_bengali_test_set
import torch
import pickle
from config import config_dict
from torch import nn


def main():
    bengali_data = BengaliData(config_dict['bengali_file_paths'])
    hindi_data = HASOCData(config_dict['hindi_file_paths'])
    with open(config_dict['bengali_file_paths']['embeddings_path'], 'rb') as f:
        bengali_embedding_weights = pickle.load(f)
    
    with open(config_dict['hindi_file_paths']['embeddings_path'], 'rb') as f:
        hindi_embedding_weights = pickle.load(f)

    ## check whether the pre-trained embeddings are the same shape as of train vocabulary
    assert bengali_embedding_weights.T.shape == (len(bengali_data.vocab), config_dict['embedding_size']), "Pre-trained Bengali embeddings size not equal to size of embedding layer"
    assert hindi_embedding_weights.T.shape == (len(hindi_data.vocab), config_dict['embedding_size']), "Pre-trained Hindi embeddings size not equal to size of embedding layer"

    ## create model instance  with configurations coming from config file
    model = SentimentNet(batch_size=config_dict['batch_size'], output_size=config_dict['out_size'], 
                            bengali_vocab_size=len(bengali_data.vocab), hidden_size=config_dict['hidden_size'], 
                            embedding_size=config_dict['embedding_size'], hindi_weights=torch.FloatTensor(hindi_embedding_weights.T), bengali_weights=torch.FloatTensor(bengali_embedding_weights.T),
                            lstm_layers=config_dict['lstm_layers'], device=config_dict['device'], hindi_vocab_size=len(hindi_data.vocab),
                            bidirectional=config_dict['is_bi_lstm'], pretrained_path=config_dict['bengali_file_paths']['pretrained_path'],
                            self_attention_config=config_dict['self_attention_config'], fc_hidden_size=config_dict['fc_hidden_size']).to(config_dict['device'])

    ## get dataloaders for train and test set
    bengali_dataloader = bengali_data.get_data_loader(batch_size=config_dict['batch_size'])
    hindi_dataloader = hindi_data.get_data_loader(batch_size=config_dict['batch_size'])

    ## filtering out embedding weights since they won't be optimized
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    
    ## training the model on train set
    """train_model(model, optimizer, bengali_dataloader=bengali_dataloader, hindi_dataloader=hindi_dataloader, 
                hindi_data=hindi_data, bengali_data=bengali_data, max_epochs=config_dict['epochs'], config_dict=config_dict)
    """
    
    ## evaluate model on test set
    evaluate_bengali_test_set(model, config_dict['bengali_model_name'], bengali_data, bengali_dataloader, device=config_dict['device'])
    evaluate_hindi_test_set(model, config_dict['hindi_model_name'], hindi_data, hindi_dataloader, device=config_dict['device'])


if __name__ == "__main__":
    main()