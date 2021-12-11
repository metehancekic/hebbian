#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"

COMMAND="python -m src.plot_neuron_activators train.regularizer=none nn.classifier=LeNet"
echo $COMMAND
eval $COMMAND

COMMAND="python -m src.plot_neuron_activators --multirun nn.implicit_normalization=l1,l2 nn.normalize_input=false,true"
echo $COMMAND
eval $COMMAND