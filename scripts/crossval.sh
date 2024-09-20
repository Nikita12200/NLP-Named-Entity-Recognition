#! /bin/bash

PYTHONPATH=. python src/crossval.py -d assets/data/brown_uts.tsv -k 5 -t kneser_nay