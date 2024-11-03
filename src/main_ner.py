from ner_tag import ner_tag
import argparse

def main():
    tagger = ner_tag()
    print('Calling train')
    tagger.train()
    print('after training is done')

if __name__ == "__main__":
    main()
