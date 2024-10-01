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
        self.unk_tag = 'UNK'
        # self.freq_threshold= 3

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
        # if self.word_to_freq_map[word.lower()] < self.freq_threshold:
        #     word = self.unk_tag
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
            # 'lemma': self.lemmatizer.lemmatize(word.lower()) if self.lemmatizer else word.lower(),  # Use lemmatizer
            # 'is_unknown': word.lower() not in self.word_to_freq_map,
        }
    
        return features

    # @staticmethod
    def sent2features(self, sentence: List[str], tags: List[str] = None) -> List[dict]:
        features = []
        prev_tag = 'START'  # Initialize previous tag as 'START' for the first word
        for i in range(len(sentence)):
            features.append(self.word2features(sentence, i, prev_tag))
        return features

    # @staticmethod
    def sent2labels(labels: List[str]) -> List[str]:
        return labels

    def train_crf(self, data: List[Tuple[List[str], List[str]]]) -> pycrfsuite.Trainer:
        print('Inside train_crf')
        trainer = pycrfsuite.Trainer(algorithm='lbfgs',verbose=False)
        count=0
        sent=0
        for sentence, tags in data:
            sent=sent+1
            for word in sentence:
                self.word_to_freq_map[word.lower()] += 1
                count=count+1
        print('total number of sentences and words in the corpus',sent,count)
        for sentence, tags in data:
            features = self.sent2features(sentence,tags)
            trainer.append(features, tags)
        
        params = {

            'c1': 1.0,
            'c2': 1e-6,
            'max_iterations': 1000,
            'feature.possible_transitions': True
        }
        
        
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

    def predict_crf(self, sentences: List[List[str]], model_path: str, beam_width: int = 2) -> List[List[str]]:
        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)
        # self.load_resources('/home/nikita/crf_pos/assignment_1/assets/freq_map.pkl', 
        #                        '/home/nikita/crf_pos/assignment_1/assets/cluster_map.pkl', 
        #                        '/home/nikita/crf_pos/assignment_1/assets/lemmatizer.pkl')
        
        # Extract possible tags from the model's state information
        possible_tags = list(tagger.labels())
        print(possible_tags)

        predictions = []

        for sentence in sentences:
            # Extract features for the entire sentence
            sentence_features = [self.word2features(sentence, i) for i in range(len(sentence))]

            # Initialize beam with (score, path)
            beam = [(0.0, [])]  # (log-probability, list of tags)

            for i in range(len(sentence)):
                new_beam = []

                # For each sequence in the beam, expand it by adding one more tag
                for score, tags in beam:
                    # if sentence[i].lower() not in self.word_to_freq_map:
                    #     # print('NOT IN DATASET',sentence[i])
                    #     new_beam.append((score, tags +['UNK']))
                    #     continue

                    for tag in possible_tags:
                        # Calculate the marginal probability of the tag at position i
                        tagger.set(sentence_features)
                        tag_score = tagger.marginal(tag, i)
                        new_score = score + tag_score

                        new_beam.append((new_score, tags + [tag]))

                # Sort new beam candidates by score and keep the top `beam_width` entries
                beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

            # The highest-scoring sequence is the first in the sorted beam
            best_score, best_tags = beam[0]
            predictions.append(best_tags)

        return predictions
 # @staticmethod
    # def predict_crf(self, sentences: List[List[str]], model_path: str) -> List[List[str]]:
    #     tagger = pycrfsuite.Tagger()
    #     tagger.open(model_path)
        
    #     predictions = []
    #     for sentence in sentences:
    #         features = self.sent2features(sentence)
    #         predicted_tags = tagger.tag(features)
    #         predictions.append(predicted_tags)
        
    #     return predictions