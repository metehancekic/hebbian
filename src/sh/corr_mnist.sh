#!/bin/bash 

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/metehan/hebbian/src/lib/"

declare -a arr=(0.00001)

for i in "${arr[@]}"
do
	COMMAND="python -m src.lib.mpl_utils.settings.py"
	echo $COMMAND
	eval $COMMAND
	COMMAND="python -m src.correlations_mnist"
	echo $COMMAND
	eval $COMMAND
done

