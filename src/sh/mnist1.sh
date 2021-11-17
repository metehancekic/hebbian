#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"

declare -a arr=(0.00001)

for i in "${arr[@]}"
do
	COMMAND="python -m src.train_mnist train.regularizer=none nn.classifier=LeNet"
	echo $COMMAND
	eval $COMMAND
	
	COMMAND="python -m src.train_mnist"
	echo $COMMAND
	eval $COMMAND
done

