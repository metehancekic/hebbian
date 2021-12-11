#!/bin/bash 

export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.attack_mnist --multirun nn.implicit_normalization=l1,l2 nn.normalize_input=false,true"
echo $COMMAND
eval $COMMAND


