import pandas as pd
import inflect
from datasets import Dataset
import re

class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = self._load_data()

    def text_cleaning(self, text):
        p = inflect.engine()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        text = ' '.join([p.number_to_words(i) if i.isdigit() else i for i in text.split()])
        return text

    def _load_data(self):
        df = pd.read_csv(self.dataset, encoding='ISO-8859-1')
        return df
    
    def save_data(self, path = 'my_data.csv'):
        self.data.to_csv(path, index=False)
        return True
    
    def clean_data(self):
        self.data = self.data[['v2', 'v1']]
        self.data = self.data.rename(columns={'v2': 'text', 'v1': 'label'})
        self.data['label'] = self.data['label'].map({'ham': 0, 'spam': 1})
        self.data['text'] = self.data['text'].apply(self.text_cleaning)
        return True
    
    def get_data(self):
        return self.data