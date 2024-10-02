from typing import List, Tuple
import pycrfsuite
import pandas as pd
import pickle
from collections import defaultdict
import nltk
nltk.download('wordnet')
nltk.data.path.append('/home/nikita/crf_pos/assignment_1/assets/data')

def load_brown_corpus(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    df = pd.read_csv(path, sep="\t")
    tokenized_text = df["tokenized_text"].apply(lambda x: x.split()).tolist()
    tokenized_pos = df["tokenized_pos"].apply(lambda x: x.split()).tolist()
    return tokenized_text, tokenized_pos

class CRFPosTagger:
    def __init__(self):
        self.word_to_freq_map = defaultdict(int)

    def save_resources(self, freq_map_path: str):
        with open(freq_map_path, 'wb') as f:
            pickle.dump(dict(self.word_to_freq_map), f) 
     

    def load_resources(self, freq_map_path: str):
        with open(freq_map_path, 'rb') as f:
            self.word_to_freq_map = pickle.load(f)

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
            'is_unknown': word.lower() not in self.word_to_freq_map,
        }
    
        return features

    def sent2features(self, sentence: List[str], tags: List[str] = None) -> List[dict]:
        features = []
        prev_tag = 'START'  # Initialize previous tag as 'START' for the first word
        for i in range(len(sentence)):
            features.append(self.word2features(sentence, i, prev_tag))
        return features

    def sent2labels(labels: List[str]) -> List[str]:
        return labels

    def train_crf(self, data: List[Tuple[List[str], List[str]]]) -> pycrfsuite.Trainer:
        print('Inside train_crf')
        trainer = pycrfsuite.Trainer(algorithm='lbfgs',verbose=False)
        #word to frequency map creation
        for sentence, tags in data:
            for word in sentence:
                self.word_to_freq_map[word.lower()] += 1
        #appending features of words to trainer object
        for sentence, tags in data:
            features = self.sent2features(sentence,tags)
            trainer.append(features, tags)
        
        params = {

            'c1': 1.0,
            'c2': 1e-6,
            'max_iterations': 1000,
            'feature.possible_transitions': True
        }
        
        #setting params
        trainer.set_params(params)
        model_path = 'assets/crf_model.crfsuite'
        trainer.train(model_path)
        print(f"Model trained and saved as '{model_path}'")
        
        # Save resources after training
        self.save_resources('assets/freq_map.pkl')
        print("Resources saved.")

    def predict_crf(self, sentences: List[List[str]], model_path: str, beam_width: int = 2) -> List[List[str]]:
        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)
        self.load_resources('/home/nikita/crf_pos/assignment_1/assets/freq_map.pkl')
        
        # Extract possible tags from the model's state information
        possible_tags = list(tagger.labels())

        predictions = []

        for sentence in sentences:
            # Extract features for the entire sentence
            sentence_features = [self.word2features(sentence, i) for i in range(len(sentence))]
            tagger.set(sentence_features)
            # Initialize beam with (score, path)
            beam = [(0.0, [])] 
            for i in range(len(sentence)):
                new_beam = []
                for score, tags in beam:
                    for tag in possible_tags:
                        # Calculate the marginal probability of the tag at position i
                        tag_score = tagger.marginal(tag, i)
                        new_score = score + tag_score
                        new_beam.append((new_score, tags + [tag]))
                        
                # Sort new beam candidates by score and keep the top `beam_width` entries
                beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]
            best_score, best_tags = beam[0]
            predictions.append(best_tags)

        return predictions
