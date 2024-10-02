#! /bin/bash

PYTHONPATH=. python src/crossval.py -d assets/data/brown_uts.tsv -k 5 --model-path assets/crf_model.crfsuite