from ner_tag import ner_tag
import argparse

def main(args: argparse.Namespace):
    tagger = ner_tag()
    print('Calling train')
    tagger.train()
    print('after training is done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-path", type=str, required=True)
    parser.add_argument("-s", "--save-path", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
