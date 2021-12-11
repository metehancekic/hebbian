#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.train_mnist --multirun nn.implicit_normalization=l2 nn.normalize_input=true train.reg.hebbian.lamda=0.5 train.reg.hebbian.k=1,2,4,8"
echo $COMMAND
eval $COMMAND


