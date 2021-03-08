from model import SentimentNet
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
from torch.autograd import Variable
from itertools import cycle
import numpy as np
logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, bengali_dataloader, hindi_dataloader, 
                bengali_data, hindi_data, max_epochs, config_dict):
    device = config_dict['device']
    criterion = nn.BCELoss() ## since we are doing binary classification
    max_accuracy_hindi = 1e-3
    max_accuracy_bengali = 1e-3
    for epoch in range(max_epochs):
        
        logging.info('Epoch: {}'.format(epoch))
        hindi_y_true = list()
        bengali_y_true = list()
        hindi_y_pred = list()
        bengali_y_pred = list()
        total_loss = 0
        for i, (hindi_train_loader, bengali_train_loader) in enumerate(zip(cycle(hindi_dataloader['train_loader']), bengali_dataloader['train_loader'])):
            hindi_batch, hindi_targets, hindi_lengths, hindi_raw_data = hindi_train_loader
            bengali_batch, bengali_targets, bengali_lengths, bengali_raw_data = bengali_train_loader

            model.zero_grad()
            hindi_batch, hindi_targets, hindi_lengths = hindi_data.sort_batch(hindi_batch, hindi_targets, hindi_lengths)## sorts the batch wrt the length of sequences
            bengali_batch, bengali_targets, bengali_lengths = bengali_data.sort_batch(bengali_batch, bengali_targets, bengali_lengths)## sorts the batch wrt the length of sequences
            
            ## perform forward pass
            ben_pred, _ = model(torch.autograd.Variable(bengali_batch).to(device), bengali_lengths.cpu().numpy(), lang='bengali') ## perform forward pass for bengali batch        
            hindi_pred, _ = model(torch.autograd.Variable(hindi_batch).to(device), hindi_lengths.cpu().numpy(), lang='hindi') ## perform forward pass for hindi batch

            ben_predictions = torch.max(ben_pred, 1)[0].float() ## get the bengali prediction values
            hindi_predictions = torch.max(hindi_pred, 1)[0].float() ## get the hindi prediction values
            
            ## since penalty coefficient is zero so we don't compute penalty attention loss here
            ## so we only compute joint loss for both bengali and hindi batches
            loss = criterion(ben_predictions.to(device), torch.autograd.Variable(bengali_targets.float()).to(device)) + \
                   criterion(hindi_predictions.to(device), torch.autograd.Variable(hindi_targets.float()).to(device))

            loss.backward() ## perform backward pass
            optimizer.step() ## update weights

            hindi_pred_idx = torch.max(hindi_pred, 1)[1] ## get pred ids
            hindi_y_true += list(hindi_targets.int().numpy()) ## accumulate targets from batch
            hindi_y_pred += list(hindi_pred_idx.data.int().detach().cpu().numpy()) ## accumulate preds from batch 
            total_loss += loss ## accumulate joint train loss

            bengali_pred_idx = torch.max(ben_pred, 1)[1] ## get pred ids
            bengali_y_true += list(bengali_targets.int().numpy()) ## accumulate targets from batch
            bengali_y_pred += list(bengali_pred_idx.data.int().detach().cpu().numpy()) ## accumulate preds from batch 
            

        hindi_acc = accuracy_score(hindi_y_true, hindi_y_pred) ## computing accuracy using sklearn's function
        bengali_acc = accuracy_score(bengali_y_true, bengali_y_pred) ## computing accuracy using sklearn's function
        
        logging.info("Train loss: {} - Hindi acc: {} -- Bengali acc: {} ".format(torch.mean(total_loss.data.float()), hindi_acc, bengali_acc))
        
        ## compute model metrics on hindi dev set
        hindi_val_acc, hindi_val_loss = evaluate_dev_set(model, hindi_data, criterion, hindi_dataloader, device, lang='hindi')
       
        ## compute model metrics on bengali dev set
        ben_val_acc, ben_val_loss = evaluate_dev_set(model, bengali_data, criterion, bengali_dataloader, device, lang='bengali')
        
        logging.info("Hindi Val loss: {} - Hindi Val acc: {} -- Bengali Val loss: {} - Bengali Val acc: {}".format(hindi_val_loss, hindi_val_acc, ben_val_loss, ben_val_acc))
        
        val_acc = np.mean(hindi_val_acc + ben_val_acc)
        if hindi_val_acc > max_accuracy_hindi:
            max_accuracy_hindi = hindi_val_acc
            logging.info('new hindi model saved') ## save the model if it is better than the prior best 
            torch.save(model.state_dict(), 'hindi_{}.pth'.format(config_dict['model_name']))
        
        if ben_val_acc > max_accuracy_bengali:
          max_accuracy_bengali = ben_val_acc
          logging.info('new bengali model saved') ## save the model if it is better than the prior best 
          torch.save(model.state_dict(), 'bengali_{}.pth'.format(config_dict['model_name']))


def attention_penalty_loss(annotation_weight_matrix, penalty_coef, device):
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

def evaluate_dev_set(model, data, criterion, data_loader, device, lang):
    """
    Evaluates the model performance on dev data
    """
    logging.info('Evaluating accuracy on dev set')

    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in data_loader['dev_loader']:
        batch, targets, lengths = data.sort_batch(batch, targets, lengths) ## sorts the batch wrt the length of sequences

        pred, annotation_weight_matrix = model(torch.autograd.Variable(batch).to(device), lengths.cpu().numpy(), lang=lang) ## perform forward pass                    
        predictions = torch.max(pred, 1)[0].float()
        pred_idx = torch.max(pred, 1)[1]
        loss = criterion(predictions.to(device), torch.autograd.Variable(targets.float()).to(device)) ## compute loss 
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int().detach().cpu().numpy())
        total_loss += loss

    acc = accuracy_score(y_true, y_pred) ## computing accuracy using sklearn's function

    return acc, torch.mean(total_loss.data.float())