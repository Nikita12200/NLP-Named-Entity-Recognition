from nltk.corpus import brown
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str)
    args = parser.parse_args()

    brown_corpus = brown.tagged_sents(tagset="universal")
    sents = list()
    tags = list()

    for idx in range(len(brown_corpus)):
        ex = list(zip(*brown_corpus[idx]))
        sents.append(" ".join(ex[0]))
        tags.append(" ".join(ex[1]))

    df = pd.DataFrame({"tokenized_text": sents, "tokenized_pos": tags})
    df.to_csv(args.output_path, sep="\t", index=None)
