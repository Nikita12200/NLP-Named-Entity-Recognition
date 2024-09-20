#! /bin/bash

PYTHONPATH=. python3 src/main_crf.py -d assets/data/brown_uts.tsv -s /home/nikita/crf_pos/assignment_1/assets/crf_model.crfsuite -t laplace
