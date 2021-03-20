from model import BengaliLSTMAttentionClassifier
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
from torch.autograd import Variable
logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict['device']
    criterion = nn.BCELoss() ## since we are doing binary classification
    max_accuracy = 1e-3
    for epoch in range(max_epochs):
        
        logging.info('Epoch: {}'.format(epoch))
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in dataloader['train_loader']:
            batch, targets, lengths = data.sort_batch(batch, targets, lengths) ## sorts the batch wrt the length of sequences

            model.zero_grad()

            pred, annotation_weight_matrix = model(torch.autograd.Variable(batch).to(device), lengths.cpu().numpy()) ## perform forward pass         
            ## compute attention loss using the following term: ||AAT - I||
            attention_loss = attention_penalty_loss(annotation_weight_matrix, 
                                                    config_dict['self_attention_config']['penalty'], device)
            pred = torch.squeeze(pred)
             ## compute combined loss: classification + attention
            loss = criterion(pred.to(device), 
                            torch.autograd.Variable(targets.float()).to(device)) + attention_loss 

            loss.backward() ## perform backward pass
            optimizer.step() ## update weights
     
           
            y_true += list(targets.int().numpy()) ## accumulate targets from batch
            pred_val = pred >= 0.5
            y_pred += list(pred_val.data.int().detach().cpu().numpy()) ## accumulate preds from batch 
            total_loss += loss ## accumulate train loss

        acc = accuracy_score(y_true, y_pred) ## computing accuracy using sklearn's function
        
        ## compute model metrics on dev set
        val_acc, val_loss = evaluate_dev_set(model, data, criterion, dataloader, device)

        if val_acc > max_accuracy:
            max_accuracy = val_acc
            logging.info('new model saved') ## save the model if it is better than the prior best 
            torch.save(model.state_dict(), '{}.pth'.format(config_dict['model_name']))
        
        logging.info("Train loss: {} - acc: {} -- Validation loss: {} - acc: {}".format(torch.mean(total_loss.data.float()), acc, val_loss, val_acc))
    return model

def attention_penalty_loss(annotation_weight_matrix, penalty_coef,device):
    """
    This function computes the loss from annotation/attention matrix
    to reduce redundancy in annotation matrix and for attention
    to focus on different parts of the sequence corresponding to the
    penalty term 'P' in the ICLR paper
    ----------------------------------
    'annotation_weight_matrix' refers to matrix 'A' in the ICLR paper
    annotation_weight_matrix shape: (batch_size, attention_out, seq_len)
    """
    batch_size, attention_out_size = annotation_weight_matrix.size(0), annotation_weight_matrix.size(1)
    ## this fn computes ||AAT - I|| where norm is the frobenius norm
    ## taking transpose of annotation matrix
    ## shape post transpose: (batch_size, seq_len, attention_out)
    annotation_weight_matrix_trans = annotation_weight_matrix.transpose(1, 2) 

    ## corresponds to AAT
    ## shape: (batch_size, attention_out, attention_out)
    annotation_mul = torch.bmm(annotation_weight_matrix, annotation_weight_matrix_trans)

    ## corresponds to 'I'
    identity = torch.eye(annotation_weight_matrix.size(1))
    ## make equal to the shape of annotation_mul and move it to device
    identity = Variable(identity.unsqueeze(0).expand(batch_size, attention_out_size, attention_out_size).to(device))
    
    ## compute AAT - I
    annotation_mul_difference = annotation_mul - identity

    ## compute the frobenius norm
    penalty = frobenius_norm(annotation_mul_difference)
    
    ## compute loss
    loss = (penalty_coef * penalty/batch_size).type(torch.FloatTensor)

    return loss
    


def frobenius_norm(annotation_mul_difference):
    """
    Computes the frobenius norm of the annotation_mul_difference input as matrix
    """
    return torch.sum(torch.sum(torch.sum(annotation_mul_difference**2,1),1)**0.5).type(torch.DoubleTensor)

def evaluate_dev_set(model, data, criterion, data_loader, device):
    """
    Evaluates the model performance on dev data
    """
    logging.info('Evaluating accuracy on dev set')

    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in data_loader['dev_loader']:
        batch, targets, lengths = data.sort_batch(batch, targets, lengths) ## sorts the batch wrt the length of sequences

        pred, annotation_weight_matrix = model(torch.autograd.Variable(batch).to(device), lengths.cpu().numpy()) ## perform forward pass                    
        pred = torch.squeeze(pred)
        loss = criterion(pred.to(device), torch.autograd.Variable(targets.float()).to(device)) ## compute loss 
        y_true += list(targets.int())
        pred_val = pred >= 0.5
        y_pred += list(pred_val.data.int().detach().cpu().numpy())
        total_loss += loss

    acc = accuracy_score(y_true, y_pred) ## computing accuracy using sklearn's function

    return acc, torch.mean(total_loss.data.float())