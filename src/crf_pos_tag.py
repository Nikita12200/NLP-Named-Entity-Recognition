from typing import List, Tuple
import pycrfsuite
import pandas as pd
import pickle

def load_brown_corpus(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    df = pd.read_csv(path, sep="\t")
    tokenized_text = df["tokenized_text"].apply(lambda x: x.split()).tolist()
    tokenized_pos = df["tokenized_pos"].apply(lambda x: x.split()).tolist()
    return tokenized_text, tokenized_pos

class CRFPosTagger:
    # Feature extraction function for each word in the sentence
    @staticmethod
    def word2features(sentence: List[str], i: int) -> dict:
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
            'capitals_inside': word[1:].lower() != word[1:]
        }
        return features

    # Extract features for each sentence
    @staticmethod
    def sent2features(sentence: List[str]) -> List[dict]:
        return [CRFPosTagger.word2features(sentence, i) for i in range(len(sentence))]

    # Convert POS tag sequence
    @staticmethod
    def sent2labels(labels: List[str]) -> List[str]:
        return labels

    # Train CRF model
    def train_crf(self, data: List[Tuple[List[str], List[str]]], smoothing: str = "kneser_nay") -> pycrfsuite.Trainer:
        print('Inside train_crf')
        trainer = pycrfsuite.Trainer(verbose=False)
        print('1')
        # Process each (sentence, tags) pair in the data list
        for sentence, tags in data:
            features = CRFPosTagger.sent2features(sentence)
            trainer.append(features, tags)
        print('2')
        # Set training parameters based on the smoothing technique
        params = {
            'c1': 1.0,   # Coefficient for L1 penalty
            'c2': 1e-3,  # Coefficient for L2 penalty
            'max_iterations': 100,
            'feature.possible_transitions': True
        }
        
        if smoothing == "laplace":
            params['c1'] = 1.0
        elif smoothing == "kneser_nay":
            params['c2'] = 1e-2  # Example adjustment for Kneser-Nay smoothing
        print('3')
        trainer.set_params(params)
        print('4')
        # Train the model
        trainer.train('/home/nikita/crf_pos/assignment_1/assets/crf_model.crfsuite')
        print("Model trained and saved as 'crf_model.crfsuite'")
        
        return trainer

    # Predict POS tags
    @staticmethod
    def predict_crf(sentences: List[List[str]], model_path: str) -> List[List[str]]:
        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)
        
        predictions = []
        for sentence in sentences:
            if not isinstance(sentence, list) or not all(isinstance(word, str) for word in sentence):
                raise ValueError("Each sentence should be a list of strings")
            features = CRFPosTagger.sent2features(sentence)
            predicted_tags = tagger.tag(features)
            predictions.append(predicted_tags)
        
        return predictions

    # Save the model using pickle
    # @staticmethod
    # def save_model(model: pycrfsuite.Trainer, model_path: str) -> None:
    #     with open(model_path, "wb") as f:
    #         pickle.dump(model, f)
    #     print(f"Model saved as {model_path}")

    # @staticmethod
    # def load_model(model_path: str) -> pycrfsuite.Tagger:
    #     with open(model_path, "rb") as f:
    #         model = pickle.load(f)
    #     return model
