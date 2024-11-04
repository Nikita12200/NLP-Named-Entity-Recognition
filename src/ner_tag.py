import numpy as np
import pickle
import nltk
from sklearn.svm import SVC
import spacy
from string import punctuation
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from datasets import load_dataset
nltk.download('stopwords')
nltk.download('wordnet')
nltk.data.path.append('assets/data')
nlp = spacy.load("en_core_web_sm")

noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", 
               "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", 
               "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ed", "ing"]
adj_suffix = [
    "able", "ese", "ful", "i", "ian", "ible", "ic", "ish", 
    "ive", "less", "ous", "ant", "ent", "al", 
    "y", "ly", "ical", "ic", "some", "like", "ous", "an", 
    "eous", "ar", "en", "ish"
]

adv_suffix = [
    "ward", "wards", "wise", "ly", 
    "fully", "lessly", "ingly", "ingly", 
    "bably", "ly", "ward", "wise", "y"
]

adjectives = [
    "happy", "sad", "large", "small", "bright", 
    "dark", "quick", "slow", "beautiful", "ugly", 
    "important", "common", "recent", "ancient", 
    "famous", "ordinary", "young", "old", 
    "reliable", "difficult"
]

# Adverbs (Not Recognized as Entities)
adverbs = [
    "quickly", "slowly", "happily", "sadly", 
    "beautifully", "easily", "often", "rarely", 
    "always", "never", "sometimes", "almost", 
    "probably", "actually", "simply", "really", 
    "particularly", "especially", "virtually", 
    "mostly"
]
class ner_tag:
    def __init__(self):
        self.SEED = 0
        self.Features_count = 18
        self.SW = stopwords.words("english")
        self.PUNCT = list(punctuation)

    def vectorize(self, w, scaled_position, prev_w=None, next_w=None):
        # print("INSDIE VECTORIZE")
        # Initialize the vector with additional features
        v = np.zeros(self.Features_count).astype(np.float32)  # +12 for new features

        # Original features
        title = 1 if w[0].isupper() else 0
        allcaps = 1 if w.isupper() else 0
        sw = 1 if w.lower() in self.SW else 0
        punct = 1 if w in self.PUNCT else 0
        word_length = len(w)
        
        # New features from word_features
        is_numeric = int(w.isdigit())
        contains_number = int(any(char.isdigit() for char in w))
        is_punctuation = int(any(char in self.PUNCT for char in w))
        has_noun_suffix = int(any(w.endswith(suffix) for suffix in noun_suffix))
        has_verb_suffix = int(any(w.endswith(suffix) for suffix in verb_suffix))
        has_adj_suffix = int(any(w.endswith(suffix) for suffix in adj_suffix))
        has_adv_suffix = int(any(w.endswith(suffix) for suffix in adv_suffix))
        
        doc = nlp(w)
        is_adj = int(doc[0].pos_ == "ADJ")  # 1 if adjective, 0 otherwise
        is_adv = int(doc[0].pos_ == "ADV") 
     
        
        # Populate the feature vector
        v[0] = title
        v[1] = allcaps
        v[2] = word_length
        v[3] = sw
        v[4] = punct
        v[5] = scaled_position
        v[6] = is_numeric
        v[7] = contains_number
        v[8] = is_punctuation
        v[9] = has_noun_suffix
        v[10] = has_verb_suffix
        v[11] = has_adj_suffix
        v[12] = has_adv_suffix
        v[13] = is_adv
        v[14] = is_adj
        # Additional features can be added as needed

        return v

    def createData(self, data, ):
        # print(data[:5])
        words = []
        features = []
        labels = []

        for d in data:
            tags = d["ner_tags"]
            tokens = d["tokens"]
            
            for i in range(len(tokens)):
                # prev_w = tokens[i - 1] if i > 0 else None
                # next_w = tokens[i + 1] if i < len(tokens) - 1 else None
                # x = self.vectorize(w=tokens[i], scaled_position=(i/len(tokens)),prev_w=prev_w, next_w=next_w)
                x = self.vectorize(w=tokens[i], scaled_position=(i/len(tokens)))

                if tags[i] <= 0:
                    y = 0
                else:
                    y = 1
                features.append(x)
                labels.append(y)
            words += tokens
        words = np.asarray(words, dtype="object")
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        # print(words[:5])
        # print(labels[:100])
        return words, features, labels
    
    def train_createData(self, data, downsample_ratio = 0.5):
        # print(data[:5])
        words = []
        features = []
        labels = []
        class_1 = 0
        class_0 = 0
        for d in data:
            tags = d["ner_tags"]
            tokens = d["tokens"]
            
            for i in range(len(tokens)):
                # if tags[i] <= 0 and np.random.rand() > downsample_ratio:
                #     continue
                # prev_w = tokens[i - 1] if i > 0 else None
                # next_w = tokens[i + 1] if i < len(tokens) - 1 else None
                # x = self.vectorize(w=tokens[i], scaled_position=(i/len(tokens)),prev_w=prev_w, next_w=next_w)
                x = self.vectorize(w=tokens[i], scaled_position=(i/len(tokens)))
                if tags[i] <= 0:
                    y = 0
                    class_0=class_0+1
                else:
                    y = 1
                    class_1=class_1+1
                features.append(x)
                labels.append(y)
            words += tokens

        words = np.asarray(words, dtype="object")
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        # print(words[:5])
        print("0 CLASSES:", class_0)
        print("1 CLASSES:",class_1)
        print(labels[:100])
        return words, features, labels
    
    def train(self):
        # data = load_dataset("Davlan/conll2003_noMISC")
        data = load_dataset("conll2003")
        data_train = data["train"]
        print("0")
        words_train, X_train, y_train = self.train_createData(data_train)
        print("00")

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        
        print("1")
        model = SVC(C=1.0, kernel='poly', degree=3, coef0=1, class_weight='balanced', random_state=self.SEED)
        print("2")
        model.fit(X_train, y_train)
        print("3")
        pickle.dump(model, open('assets/num/nei_model.sav', 'wb'))
        pickle.dump(scaler, open('assets/num/scaler_model.sav', 'wb'))

    def infer(self, sentence, model_path, scaler_path,flag ):
        # Load the NEI model and scaler
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        if hasattr(model, 'coef_'):
            print("Feature weights:")
            print(model.coef_)

        # Optionally, you can also print feature importance
        # feature_importance = np.abs(model.coef_[0])  # For binary classification
        # feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
        # for name, weight in zip(feature_names, feature_importance):
        #     print(f"{name}: {weight:.4f}")
        # Tokenize and extract features
        if(flag == 0):
            tokens = word_tokenize(sentence)
        else:
            tokens = sentence
            
        # print(tokens[:100])
        features = [self.vectorize(w=tokens[i], scaled_position=(i / len(tokens))) for i in range(len(tokens))]
        print(len(features[0]))
        features = np.asarray(features, dtype=np.float32)
        print(features.shape)
        print(features)
        print('AFTER:')
        scaled_features = scaler.transform(features)
        print(scaled_features)
        # Make predictions
        pred = model.predict(scaled_features).astype(int)
        
        return pred, tokens

