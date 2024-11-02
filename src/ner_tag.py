import numpy as np
import pickle
import nltk
from sklearn.svm import SVC

from string import punctuation
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from datasets import load_dataset
nltk.download('stopwords')
nltk.download('wordnet')
nltk.data.path.append('assets/data')

class ner_tag:
    def __init__(self):
        self.SEED = 0
        self.Features_count = 6
        self.SW = stopwords.words("english")
        self.PUNCT = list(punctuation)

    def vectorize(self, w, scaled_position,prev_w=None, next_w=None):
        v = np.zeros(self.Features_count + 2).astype(np.float32)  # +2 for prev_title and next_title
        title = 1 if w[0].isupper() else 0
        allcaps = 1 if w.isupper() else 0
        sw = 1 if w.lower() in self.SW else 0
        punct = 1 if w in self.PUNCT else 0

        # Add new features for previous and next words
        prev_title = 1 if prev_w and prev_w[0].isupper() else 0
        next_title = 1 if next_w and next_w[0].isupper() else 0

        return [title, allcaps, len(w), sw, punct, scaled_position, prev_title, next_title]

    def createData(self, data):
        # print(data[:5])
        words = []
        features = []
        labels = []
        for d in data:
            tags = d["ner_tags"]
            tokens = d["tokens"]
            
            for i in range(len(tokens)):
                prev_w = tokens[i - 1] if i > 0 else None
                next_w = tokens[i + 1] if i < len(tokens) - 1 else None
                x = self.vectorize(w=tokens[i], scaled_position=(i/len(tokens)),prev_w=prev_w, next_w=next_w)
                y = 0 if tags[i] == "O" else 1
                features.append(x)
                labels.append(y)
            words += tokens
        words = np.asarray(words, dtype="object")
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        # print(words[:5])
        print(labels[:5])
        return words, features, labels
    

    def train(self):
        data = load_dataset("Davlan/conll2003_noMISC")
        data_train = data["train"]

        words_train, X_train, y_train = self.createData(data_train)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        model = SVC(C=10.0, kernel="rbf",gamma=0.1, class_weight="balanced", random_state=self.SEED, verbose=True)
        model.fit(X_train, y_train)

        pickle.dump(model, open('assets/nei_model.sav', 'wb'))
        pickle.dump(scaler, open('assets/scaler_model.sav', 'wb'))

    def infer(self, sentence, model_path, scaler_path,flag ):
        # Load the NEI model and scaler
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # Tokenize and extract features
        if(flag == 0):
            tokens = word_tokenize(sentence)
        else:
            tokens = sentence
            
        print(tokens)
        features = [self.vectorize(w=tokens[i], scaled_position=(i / len(tokens))) for i in range(len(tokens))]
        features = np.asarray(features, dtype=np.float32)
        scaled_features = scaler.transform(features)

        # Make predictions
        pred = model.predict(scaled_features).astype(int)
        
        return pred, tokens

