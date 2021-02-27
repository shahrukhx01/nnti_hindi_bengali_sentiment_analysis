from data import HASOCData
from model import HindiLSTMClassifier
from train import train_model
import torch
import pickle
from config import config_dict
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    data = HASOCData(config_dict['file_paths'])
    with open(config_dict['file_paths']['embeddings_path'], 'rb') as f:
        embedding_weights = pickle.load(f)

    ## check whether the pre-trained embeddings are the same shape as of train vocabulary
    assert embedding_weights.T.shape == (len(data.vocab), config_dict['embedding_size']), "Pre-trained embeddings size not equal to size of embedding layer"

    model = HindiLSTMClassifier(batch_size=config_dict['batch_size'], output_size=config_dict['num_classes'], vocab_size=len(data.vocab), \
                                hidden_size=config_dict['hidden_size'], embedding_size=config_dict['embedding_size'], weights=torch.FloatTensor(embedding_weights.T),\
                                lstm_layers=config_dict['lstm_layers'], device=config_dict['device'])
                                
    #model.load_state_dict(torch.load('hindi_classifier.pth'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    train_model(model, optimizer, data, config_dict['batch_size'], max_epochs=config_dict['epochs'], config_dict=config_dict)



if __name__ == "__main__":
    main()