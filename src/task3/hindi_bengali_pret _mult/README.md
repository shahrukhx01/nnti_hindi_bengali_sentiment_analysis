# Multi Task Learning with Self Attentive Sentence Embeddings

## Directory structure

* `data` directory contains stopwords and bengali subset dataset files.
* `artefacts` contains checkpointed/saved word embeddings and PyTorch models which performed best on development set for different hyper-parameter configurations.
* `Bengali_BiLSTM_Attention_Binary_Classifier.ipynb` contains the runtime code for training classifier on train set and evaluating on test set.
* `bengali_dataset.py` extends standard PyTorch dataset utility to conveniently batch data.
* `bengali_subset_data.py` script for splitting Bengali dataset which is roughly equal to Hindi dataset.
* `config.py` python script which exports all configurations for hyper-parameters, saving and loading models as python dictionary.
* `data.py` main script for calling preprocssing routine, transforming text to tensors, wrapping tensors with Pytorch dataset class, then creating train, test and dev dataloaders.
* `eval.py` evaluates the given model on test dataset.
* `main.py` alternate to `Bengali_BiLSTM_Attention_Binary_Classifier.ipynb` if you prefer shell based training/testing of model.
* `model.py` contains the main structure for binary classifier using PyTorch's `nn.Module` class to create layers and implement forward pass.
* `preprocess.py` implements the data preprocessing pipelines for text field and returns the dataframe by adding extra column as `clean_text` to it.
* `train.py` responsible for training the model and monitoring development set accuracy for checkpointing model after each epoch.
