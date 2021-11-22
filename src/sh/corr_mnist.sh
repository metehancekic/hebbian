#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"


COMMAND="python -m src.correlations_mnist train.regularizer=hebbian_1.0 nn.classifier=T_LeNet"
echo $COMMAND
eval $COMMAND




