#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"

declare -a arr=(0.00001)

for i in "${arr[@]}"
do
	# COMMAND="python -m src.plot_activations train.regularizer=hebbian_1.0 nn.classifier=T_LeNet"
	# echo $COMMAND
	# eval $COMMAND

	COMMAND="python -m src.plot_activations train.regularizer=hebbian_1.0 nn.classifier=Dn_LeNet"
	echo $COMMAND
	eval $COMMAND
	
done

