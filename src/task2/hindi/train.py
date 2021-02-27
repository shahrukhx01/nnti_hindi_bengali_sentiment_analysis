from model import HindiLSTMClassifier
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_model(model, optimizer, hasoc_dataloader, data, max_epochs, config_dict):
    device = config_dict['device']
    criterion = nn.BCELoss()
    max_accuracy = 7e-1
    for epoch in range(max_epochs):
        
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in hasoc_dataloader['train_loader']:
            batch, targets, lengths = data.sort_batch(batch, targets, lengths)

            model.zero_grad()

            pred = model(torch.autograd.Variable(batch).to(device), lengths.cpu().numpy())            
            predictions = torch.max(pred, 1)[0].float()
            loss = criterion(predictions.to(device), torch.autograd.Variable(targets.float()).to(device))

            loss.backward()
            optimizer.step()
     
            pred_idx = torch.max(pred, 1)[1]
           
            y_true += list(targets.int().numpy())
            
            y_pred += list(pred_idx.data.int().detach().cpu().numpy())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        

        if acc > max_accuracy:
            max_accuracy = acc
            print('new model saved with epoch accuracy {}'.format(max_accuracy))
            torch.save(model.state_dict(), '{}.pth'.format(config_dict['model_name']))
        else:
            print('Epoch accuracy {}'.format(acc))
        
        print("Train loss: {} - acc: {}".format(torch.mean(total_loss.data.float()), acc))
    return model
