import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
logging.basicConfig(level=logging.INFO)


def evaluate_test_set(model, data, data_loader, device):
    model.eval()
    logging.info('Evaluating accuracy on test set')

    target_names = ['negative sentiment', 'positive sentiment']
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in data_loader['test_loader']:
        batch, targets, lengths = data.sort_batch(batch, targets, lengths)

        pred = model(torch.autograd.Variable(batch).to(device), lengths.cpu().numpy())            
        predictions = torch.max(pred, 1)[0].float()
        pred_idx = torch.max(pred, 1)[1]

        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int().detach().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)

    print("Test acc: {}".format(acc))
    print('\n\n')
    print(classification_report(y_true, y_pred, target_names=target_names))