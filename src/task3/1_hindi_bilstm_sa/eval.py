import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
logging.basicConfig(level=logging.INFO)

"""
Script for evaluating the neural network on test set
"""

def evaluate_test_set(model, data, data_loader, device):
    """
    Evaluates the model performance on test data
    """
    model.eval()
    logging.info('Evaluating accuracy on test set')

    target_names = ['non hate speech', 'hate speech']
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in data_loader['test_loader']:
        batch, targets, lengths = data.sort_batch(batch, targets, lengths) ## sorts the batch wrt the length of sequences
        pred, annotation_weight_matrix = model(torch.autograd.Variable(batch).to(device), lengths.cpu().numpy()) ## perform forward pass                    
       
        pred = torch.squeeze(pred)

        y_true += list(targets.int())
        pred_val = pred >= 0.5
        y_pred += list(pred_val.data.int().detach().cpu().numpy())

        ## commented script for visualizing attention in sequences
        """data_out = []
        for i,x in enumerate(list(predictions.data.float().detach().cpu().numpy())):
           if x > 0.9 and y_pred[i] == 1:
               seq = []
               for x in list(batch[:,i].data.int().detach().cpu().numpy()):
                   if x == 0:
                       break
                   else:
                        seq.append(data.index2word[x])
               for k in list(annotation_weight_matrix[i, :, :].data.float().detach().cpu().numpy()):
                    data_in = []
                    for idx, tok in enumerate(seq):
                       
                        data_in.append({tok: k[idx]})
                    
                    data_out.append(data_in)
            
               #print(" ".join(seq), i)

        #print(data_out)
        with open("attn.txt", "a+") as output:
            output.write(str(data_out))
        break"""

    acc = accuracy_score(y_true, y_pred) ## computing accuracy using sklearn's function

    print("Test acc: {}".format(acc))
    print('\n\n')
    print(classification_report(y_true, y_pred, target_names=target_names)) ## computing other classification metrics via sklearn in classification report