from data import HASOCData
from model import HindiLSTMClassifier
from train import train_model 
from eval import evaluate_test_set
import torch
import pickle
from config import config_dict
from torch import nn

def main():
    data = HASOCData(config_dict['file_paths'])
    with open(config_dict['file_paths']['embeddings_path'], 'rb') as f:
        embedding_weights = pickle.load(f)

    ## check whether the pre-trained embeddings are the same shape as of train vocabulary
    assert embedding_weights.T.shape == (len(data.vocab), config_dict['embedding_size']), "Pre-trained embeddings size not equal to size of embedding layer"

    model = HindiLSTMClassifier(batch_size=config_dict['batch_size'], output_size=config_dict['num_classes'], 
                                vocab_size=len(data.vocab), hidden_size=config_dict['hidden_size'], 
                                embedding_size=config_dict['embedding_size'], weights=torch.FloatTensor(embedding_weights.T),
                                lstm_layers=config_dict['lstm_layers'], device=config_dict['device']).to(config_dict['device'])

    

    ## get dataloaders for train and test set
    hasoc_dataloader = data.get_data_loader(batch_size=config_dict['batch_size'])

    ## filtering out embedding weights since they won't be optimized
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    ## training the model on train set
    train_model(model, optimizer, hasoc_dataloader, data, max_epochs=config_dict['epochs'],config_dict=config_dict)

    ## loading the best model saved during training from disk
    model.load_state_dict(torch.load('{}.pth'.format(config_dict['model_name'])))

    ## evaluate model on test set
    evaluate_test_set(model, data, hasoc_dataloader, device=config_dict['device'])

if __name__ == "__main__":
    main()