#! /bin/bash

PYTHONPATH=. streamlit run src/demo_ner.py -- --nei-model-path assets/nei_model.sav --scaler-model-path assets/scaler_model.sav
