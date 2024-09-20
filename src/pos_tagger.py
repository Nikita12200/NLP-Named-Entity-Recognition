from typing import List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
import pickle


def load_brown_corpus(path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path, sep="\t")
    tokenized_text = df["tokenized_text"].tolist()
    tokenized_pos = df["tokenized_pos"].tolist()
    return tokenized_text, tokenized_pos


from typing import List, Tuple, Optional
from collections import Counter
import numpy as np


class HMMPosTagger:
    def __init__(self, vocab_frac: float = 0.7):
        self.emission = None
        self.transition = None
        self.initial = None
        self.vocab_frac = vocab_frac
        self._trained = False
        self.unk_word = "<unk>"

    def _build_vocab(self, data: List[Tuple[str, str]]) -> None:
        tags = set()
        wordcount = Counter()

        for wordseq, tagseq in data:
            wordseq, tagseq = self.preprocess((wordseq, tagseq))
            tags.update(tagseq)
            wordcount.update(wordseq)

        self.words = [
            word
            for word, _ in wordcount.most_common(int(len(wordcount) * self.vocab_frac))
        ]
        self.words.append(self.unk_word)
        self.tags = list(tags)

        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.tags)}
        self.word2idx = {tag: idx for idx, tag in enumerate(self.words)}
        self.idx2word = {idx: tag for idx, tag in enumerate(self.words)}

    def preprocess(
        self,
        datapoint: Tuple[str, Optional[str]],
        return_idx: bool = False,
    ) -> Tuple[List[int], Optional[List[int]]]:
        wordseq = [word for word in datapoint[0].lower().split()]
        if return_idx:
            wordseq = [
                self.word2idx.get(word, self.word2idx[self.unk_word])
                for word in wordseq
            ]

        if len(datapoint) == 2:
            tagseq = [tag for tag in datapoint[1].lower().split()]
            if return_idx:
                tagseq = [self.tag2idx.get(tag) for tag in tagseq]
            return (wordseq, tagseq)
        return (wordseq,)

    def postprocess(
        self, datapoint: Tuple[List[int], Optional[List[int]]]
    ) -> Tuple[str, Optional[str]]:
        words = " ".join([self.idx2word.get(wordidx) for wordidx in datapoint[0]])
        tags = " ".join([self.idx2tag.get(tagidx) for tagidx in datapoint[1]])
        return (words, tags)

    def train(self, data: List[Tuple[str, str]], smoothing: str = "laplace") -> None:
        self._build_vocab(data)

        transition_count = np.zeros((len(self.tags), len(self.tags)))
        emission_count = np.zeros((len(self.tags), len(self.words)))
        initial_count = np.zeros((len(self.tags)))

        for wordseq, tagseq in data:
            wordseq, tagseq = self.preprocess((wordseq, tagseq), return_idx=True)

            if len(wordseq) != len(tagseq):
                continue

            for idx in range(len(wordseq)):
                if idx == 0:
                    initial_count[tagseq[idx]] += 1
                if idx < len(wordseq) - 1:
                    transition_count[tagseq[idx], tagseq[idx + 1]] += 1
                emission_count[tagseq[idx], wordseq[idx]] += 1

        if smoothing == "kneser_ney":
            # Kneser-Ney Smoothing
            discount = 0.75

            continuation_transition = np.sum(transition_count > 0, axis=0)
            continuation_emission = np.sum(emission_count > 0, axis=0)

            transition_probs = np.maximum(
                transition_count - discount, 0
            ) / transition_count.sum(axis=1).reshape(-1, 1)
            transition_backoff = (
                discount * continuation_transition / continuation_transition.sum()
            )
            self.transition = transition_probs + transition_backoff.reshape(-1, 1)

            emission_probs = np.maximum(
                emission_count - discount, 0
            ) / emission_count.sum(axis=1).reshape(-1, 1)
            emission_backoff = (
                discount * continuation_emission / continuation_emission.sum()
            )
            self.emission = emission_probs + emission_backoff.reshape(-1, 1)

            self.initial = (initial_count + 1) / (initial_count.sum() + len(self.tags))
        else:
            # Laplace (Add-One) Smoothing
            self.transition = (transition_count + 1) / (
                transition_count.sum(axis=1).reshape(-1, 1) + len(self.tags)
            )
            self.emission = (emission_count + 1) / (
                emission_count.sum(axis=1).reshape(-1, 1) + len(self.words)
            )
            self.initial = (initial_count + 1) / (initial_count.sum() + len(self.tags))

        self._trained = True

    def predict(self, wordseq: str) -> str:
        if not self._trained:
            raise ValueError("The model must be trained before predicting.")

        datapoint = self.preprocess((wordseq,), return_idx=True)
        wordseq = datapoint[0]
        n_tags = len(self.tags)
        n_words = len(wordseq)

        viterbi = np.zeros((n_tags, n_words))
        backpointer = np.zeros((n_tags, n_words), dtype=int)

        for s in range(n_tags):
            viterbi[s, 0] = self.initial[s] * self.emission[s, wordseq[0]]
            backpointer[s, 0] = 0

        for t in range(1, n_words):
            for s in range(n_tags):
                max_prob = -1
                best_prev_state = 0
                for s_prime in range(n_tags):
                    prob = (
                        viterbi[s_prime, t - 1]
                        * self.transition[s_prime, s]
                        * self.emission[s, wordseq[t]]
                    )
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = s_prime
                viterbi[s, t] = max_prob
                backpointer[s, t] = best_prev_state

        best_last_state = np.argmax(viterbi[:, n_words - 1])
        tagseq = [best_last_state]

        for t in range(n_words - 1, 0, -1):
            tagseq.insert(0, backpointer[tagseq[0], t])

        wordseq, tagseq = self.postprocess((wordseq, tagseq))
        return tagseq

    def save(self, path: str) -> None:
        with open(path, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(self, path: str) -> "HMMPosTagger":
        with open(path, "rb") as fp:
            return pickle.load(fp)
