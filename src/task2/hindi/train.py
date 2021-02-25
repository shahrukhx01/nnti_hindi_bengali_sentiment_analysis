from data import HASOCData
from model import HindiLSTMClassifier
import torch
import pickle
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def forward_pass(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss

def train_model(model, optimizer, data, batch_size, max_epochs):
    criterion = nn.NLLLoss(size_average=False)
    max_accuracy = 5e-1
    for epoch in range(max_epochs):
        
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in data.get_data_loader(batch_size=batch_size):
            batch, targets, lengths = data.sort_batch(batch, targets, lengths)
            model.zero_grad()
            pred, loss = forward_pass(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()
            
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        if acc > max_accuracy:
            max_accuracy = acc
            print('new model saved with epoch accuracy {}'.format(max_accuracy))
            torch.save(model.state_dict(), '/content/hindi_classifier.pth')
        else:
            print('Epoch accuracy'.format(acc))
        """val_loss, val_acc = evaluate_validation_set(model, dev, x_to_ix, y_to_ix, criterion)
        print("Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(total_loss.data.float()/len(train), acc,
                                                                                val_loss, val_acc))"""
    return model



def main():
   FILE_PATHS = {'data_file': '/Users/shahrukh/Desktop/NNTI-WS2021-NLP-Project-main/data/hindi_hatespeech.tsv',\
       'stpwds_file':'/Users/shahrukh/Desktop/NNTI-WS2021-NLP-Project-main/data/stopwords-hi.txt',\
       'embeddings_path':'/Users/shahrukh/Desktop/NNTI-WS2021-NLP-Project-main/src/artefacts/embedding_weights_all_sample.pickle'}
   data = HASOCData(FILE_PATHS)
   batch_size = 32
   num_classes = 2
   vocab_size = len(data.vocab)
   
   
   #train_iter = data.get_data_loader(batch_size=batch_size)
   with open(FILE_PATHS['embeddings_path'], 'rb') as f:
       embedding_weights = pickle.load(f)


   print('embed',embedding_weights.T.shape, 'expect', (vocab_size, 300))
   model = HindiLSTMClassifier(batch_size=batch_size, output_size=num_classes, vocab_size=vocab_size, \
                                hidden_size=32, embedding_size=300, weights=torch.FloatTensor(embedding_weights.T),\
                                lstm_layers=2)

   optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

   train_model(model, optimizer, data, batch_size, max_epochs=5)



if __name__ == "__main__":
    main()