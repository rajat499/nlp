#!/usr/bin/env bash

if [[ "$1" == "train" ]]
then
  python train.py $2 $3 cs5170415_model
elif [[ "$1" == "test" ]]
then
  python test.py $2 $3 cs5170415_model
fi