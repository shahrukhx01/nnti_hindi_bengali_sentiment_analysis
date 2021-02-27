from data import HASOCData
from model import HindiLSTMClassifier
import torch
import pickle
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_model(model, optimizer, data, batch_size, max_epochs, config_dict):
    criterion = nn.BCELoss()
    max_accuracy = 6e-1
    for epoch in range(max_epochs):
        
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        hasoc_dataloader = data.get_data_loader(batch_size=batch_size)
        for batch, targets, lengths, raw_data in hasoc_dataloader['train_loader']:
            batch, targets, lengths = data.sort_batch(batch, targets, lengths)

            model.zero_grad()

            pred = model(torch.autograd.Variable(batch).to(config_dict['device']), lengths.cpu().numpy())            
            predictions = torch.max(pred, 1)[0].float()
            loss = criterion(predictions, torch.autograd.Variable(targets.float()))

            loss.backward()
            optimizer.step()
     
            pred_idx = torch.max(pred, 1)[1]
           
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        test_loss, test_acc = evaluate_validation_set(model, data, hasoc_dataloader, criterion)

        if test_acc > max_accuracy:
            max_accuracy = test_acc
            print('new model saved with epoch accuracy {}'.format(max_accuracy))
            torch.save(model.state_dict(), '{}.pth'.format(config_dict['model_name']))
        else:
            print('Epoch accuracy {}'.format(acc))
        
        print("Train loss: {} - acc: {} \Test loss: {} - acc: {}".format(torch.mean(total_loss.data.float()), acc,
                                                                                test_loss, test_acc))
    return model



def evaluate_validation_set(model, data, data_loader, criterion):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in data_loader['test_loader']:
        batch, targets, lengths = data.sort_batch(batch, targets, lengths)
        pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())            
        predictions = torch.max(pred, 1)[0].float()
        loss = criterion(predictions, torch.autograd.Variable(targets.float()))
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    return torch.mean(total_loss.data.float()), acc