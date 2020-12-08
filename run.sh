#!/bin/bash

cd data
unzip train_preliminary.zip
unzip train_semi_final.zip
unzip test.zip
cd ..

python src/preprocess.py
python src/get_seqs.py
python src/w2v.py

python src/get_stat_feas.py
python src/get_lgb_feas.py
python src/get_feas.py

python src/lstm0.py
python src/lstm1.py
python src/lstm2.py
python src/lstm3.py