import re

class Preprocess:
    def __init__(self, stpwds_file_path):
        self.USERNAME_PATTERN = r'@([A-Za-z0-9_]+)'
        self.PUNCTUATION_PATTERN = '\'’|!@$%^&*()_+<>?:.,;-'
        self.STOPWORDS_PATH = stpwds_file_path
        self.load_stopwords()
    
    def load_stopwords(self):
        stopwords_hindi_file = open(self.STOPWORDS_PATH, 'r')
        self.stopwords_hindi = [line.replace('\n','') for line in stopwords_hindi_file.readlines()]


    def remove_punctuations(self, text):
        return "".join([c for c in text if c not in self.PUNCTUATION_PATTERN])
    
    def remove_stopwords(self, text):
        return " ".join([word for word in text.split() if word not in self.stopwords_hindi])
    
    def remove_usernames(self, text):  
        return re.sub(self.USERNAME_PATTERN, '', text)
    
    def perform_preprocessing(self, data):       
        data['clean_text'] = data.text.apply(lambda text: text.lower()) ## normalizing text to lower case
        data['clean_text'] = data.clean_text.apply(self.remove_usernames)## removing usernames
        data['clean_text'] = data.clean_text.apply(self.remove_punctuations)## removing punctuations
        data['clean_text'] = data.clean_text.apply(self.remove_stopwords)## removing stopwords

        return data