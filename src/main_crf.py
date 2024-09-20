from src.crf_pos_tag import CRFPosTagger, load_brown_corpus
import argparse

def main(args: argparse.Namespace):
    wordseqs, tagseqs = load_brown_corpus(args.dataset_path)
    tagger = CRFPosTagger()
    print('Calling train_Crf')
    tagger.train_crf(list(zip(wordseqs, tagseqs)), smoothing=args.smoothing_technique)
    print('Calling save_model')
    # tagger.save_model(tagger, args.save_path)
    print('After all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-path", type=str, required=True)
    parser.add_argument("-s", "--save-path", type=str, required=True)
    parser.add_argument(
        "-t",
        "--smoothing-technique",
        choices=["laplace", "kneser_nay"],
        default="kneser_nay",
    )
    args = parser.parse_args()
    main(args)
