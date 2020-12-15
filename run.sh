#!/bin/bash

rm -rf res/pickle
mkdir res/pickle
python3 src/extract_data.py $1
python3 src/search_data.py $2
python3 src/prec_recall.py res/ground-truth-unique.txt