#!/usr/bin/env bash
pip install -r requirements.txt
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install torchtext==0.7.0
python -m spacy download en
cp -r /scratch/cse/phd/csz198394/A1.2/cs5170415/.vector_cache ./
cp -r /scratch/cse/phd/csz198394/A1.2/cs5170415/text_field ./
cp -r /scratch/cse/phd/csz198394/A1.2/cs5170415/cs5170415_model.pth ./
