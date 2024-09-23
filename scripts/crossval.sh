#! /bin/bash

PYTHONPATH=. python src/crossval.py -d assets/data/brown_uts.tsv -k 5 --model-path /home/nikita/crf_pos/assignment_1/assets/crf_model.crfsuite -t kneser_nay