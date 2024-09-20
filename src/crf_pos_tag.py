from typing import List, Tuple
import pycrfsuite
import pandas as pd
import pickle
from collections import defaultdict
import nltk
nltk.download('wordnet')
nltk.data.path.append('/home/nikita/crf_pos/assignment_1/assets/data')
from nltk.stem import WordNetLemmatizer  # You may need to install nltk

# Make sure to download the WordNet corpus if using NLTK:
# import nltk
# nltk.download('wordnet')

def load_brown_corpus(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    df = pd.read_csv(path, sep="\t")
    tokenized_text = df["tokenized_text"].apply(lambda x: x.split()).tolist()
    tokenized_pos = df["tokenized_pos"].apply(lambda x: x.split()).tolist()
    return tokenized_text, tokenized_pos

class CRFPosTagger:
    def __init__(self):
        self.word_to_freq_map = defaultdict(int)
        self.word_to_cluster_map = {}
        self.lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer

    def save_resources(self, freq_map_path: str, cluster_map_path: str, lemmatizer_path: str):
        with open(freq_map_path, 'wb') as f:
            pickle.dump(dict(self.word_to_freq_map), f)  # Convert defaultdict to dict
        with open(cluster_map_path, 'wb') as f:
            pickle.dump(self.word_to_cluster_map, f)
        with open(lemmatizer_path, 'wb') as f:
            pickle.dump(self.lemmatizer, f)

    def load_resources(self, freq_map_path: str, cluster_map_path: str, lemmatizer_path: str):
        with open(freq_map_path, 'rb') as f:
            self.word_to_freq_map = pickle.load(f)
        with open(cluster_map_path, 'rb') as f:
            self.word_to_cluster_map = pickle.load(f)
        with open(lemmatizer_path, 'rb') as f:
            self.lemmatizer = pickle.load(f)

    def word2features(self, sentence: List[str], i: int,prev_tag: str = None) -> dict:
        word = sentence[i]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'prefix_1': word[:1],
            'prefix_2': word[:2],
            'prefix_3': word[:3],
            'prefix_4': word[:4],
            'suffix_1': word[-1:],
            'suffix_2': word[-2:],
            'suffix_3': word[-3:],
            'suffix_4': word[-4:],
            'is_first': i == 0,
            'is_last': i == len(sentence) - 1,
            'is_first_capital': word[0].isupper(),
            'is_all_caps': word.isupper(),
            'is_all_lower': word.islower(),
            'prev_word': '' if i == 0 else sentence[i - 1].lower(),
            'next_word': '' if i == len(sentence) - 1 else sentence[i + 1].lower(),
            'has_hyphen': '-' in word,
            'is_numeric': word.isdigit(),
            'capitals_inside': word[1:].lower() != word[1:],
            'word_freq': self.word_to_freq_map.get(word.lower(), 0),  # Use frequency map
            'word_cluster': self.word_to_cluster_map.get(word.lower(), 'UNK'),  # Use cluster map
            'lemma': self.lemmatizer.lemmatize(word.lower()) if self.lemmatizer else word.lower(),  # Use lemmatizer
            'prev_tag': prev_tag if prev_tag else 'START'
        }
        return features

    # @staticmethod
    def sent2features(self, sentence: List[str], tags: List[str] = None) -> List[dict]:
        features = []
        prev_tag = 'START'  # Initialize previous tag as 'START' for the first word
        for i in range(len(sentence)):
            # Check if tags are provided (for training). If not, continue using 'START' or the previous tag.
            if tags:
                prev_tag = tags[i - 1] if i > 0 else 'START'  # Use tags only if they are available
            features.append(self.word2features(sentence, i, prev_tag))
        return features

    # @staticmethod
    def sent2labels(labels: List[str]) -> List[str]:
        return labels

    def train_crf(self, data: List[Tuple[List[str], List[str]]], smoothing: str = "kneser_nay") -> pycrfsuite.Trainer:
        print('Inside train_crf')
        trainer = pycrfsuite.Trainer(verbose=False)
        
        for sentence, tags in data:
            for word in sentence:
                # Update frequency map
                self.word_to_freq_map[word.lower()] += 1
                # Update cluster map (simple example, replace with actual clustering logic)
                self.word_to_cluster_map[word.lower()] = hash(word) % 10  # Simple hash for clustering
            
            features = self.sent2features(sentence,tags)
            trainer.append(features, tags)
        
        params = {
            'c1': 1.0,
            'c2': 1e-3,
            'max_iterations': 100,
            'feature.possible_transitions': True
        }
        
        if smoothing == "laplace":
            params['c1'] = 1.0
        elif smoothing == "kneser_nay":
            params['c2'] = 1e-2
        
        trainer.set_params(params)
        # 
        model_path = '/home/nikita/crf_pos/assignment_1/assets/crf_model.crfsuite'
        trainer.train(model_path)
        print(f"Model trained and saved as '{model_path}'")
        
        # Save resources after training
        self.save_resources('/home/nikita/crf_pos/assignment_1/assets/freq_map.pkl', 
                            '/home/nikita/crf_pos/assignment_1/assets/cluster_map.pkl', 
                            '/home/nikita/crf_pos/assignment_1/assets/lemmatizer.pkl')
        print("Resources saved.")
    # @staticmethod
    def predict_crf(self, sentences: List[List[str]], model_path: str) -> List[List[str]]:
        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)
        
        predictions = []
        for sentence in sentences:
            features = self.sent2features(sentence)
            predicted_tags = tagger.tag(features)
            predictions.append(predicted_tags)
        
        return predictions

# Example usage
crf_tagger = CRFPosTagger()

# Load your resources (frequency map, cluster map, lemmatizer) if available
try:
    crf_tagger.load_resources('/home/nikita/crf_pos/assignment_1/assets/freq_map.pkl', 
                               '/home/nikita/crf_pos/assignment_1/assets/cluster_map.pkl', 
                               '/home/nikita/crf_pos/assignment_1/assets/lemmatizer.pkl')
except FileNotFoundError:
    print("Resource files not found. They will be created during training.")

# Train your model with your training data
# data should be in the format: List[Tuple[List[str], List[str]]]
# where each Tuple contains (sentence, tags)
# crf_tagger.train_crf(data)

# Predict using the trained model
# sentences should be in the format: List[List[str]]
# predictions = crf_tagger.predict_crf(sentences, '/home/nikita/crf_pos/assignment_1/assets/crf_model.crfsuite')
