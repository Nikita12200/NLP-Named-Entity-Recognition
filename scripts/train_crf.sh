#! /bin/bash

PYTHONPATH=. python3 src/main_crf.py -d assets/data/brown_uts.tsv -s assets/crf_model.crfsuite -t laplace
