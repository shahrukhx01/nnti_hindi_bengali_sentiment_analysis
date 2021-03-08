# Joint Dual Input Learning with Self Attentive Sentence Embeddings
## Directory structure

* `data` directory contains stopwords and bengali subset dataset files.
* `artefacts` contains checkpointed/saved word embeddings and PyTorch models which performed best on development set for different hyper-parameter configurations.
* `Sentiment_Net.ipynb.ipynb` contains the runtime code for training classifier on train set and evaluating on test set.
* `bengali_dataset.py` & `hindi_dataset.py` extends standard PyTorch dataset utility to conveniently batch data.
* `bengali_subset_data.py` script for splitting Bengali dataset which is roughly equal to Hindi dataset.
* `config.py` python script which exports all configurations for hyper-parameters, saving and loading models as python dictionary.
* `hindi_data.py` & `bengali_data.py` main script for calling preprocssing routine, transforming text to tensors, wrapping tensors with Pytorch dataset class, then creating train, test and dev dataloaders for Hindi and Bengali.
* `hindi_eval.py` & `bengali_eval.py` evaluates the given model on hindi test and bengali test dataset respectively.
* `main.py` alternate to `Sentiment_Net.ipynb.ipynb` if you prefer shell based training/testing of model.
* `model.py` contains the main structure for binary classifier using PyTorch's `nn.Module` class to create layers and implement forward pass.
* `hindi_preprocess.py` & `bengali_preprocess.py` implements the data preprocessing pipelines for text field and returns the dataframe by adding extra column as `clean_text` to it.
* `train.py` responsible for training the model and monitoring development set accuracy for checkpointing model after each epoch.

![Joint Dual Input Learning Ideation](https://github.com/shahrukhx01/nnti_hindi_bengali_sentiment_analysis/blob/main/src/task3/3_hindi_bengali_bilstm_sa_jdil/sentiment_net_ideation.png)
