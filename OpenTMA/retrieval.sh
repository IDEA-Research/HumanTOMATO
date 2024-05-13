#!/bin/bash

path1="./experiments/temos/H3D-TMR-v1/embeddings/val/epoch_99/"
path2="./experiments/temos/H3D-TMR-v1/embeddings/val/epoch_599/"
path3="./experiments/temos/H3D-TMR-v1/embeddings/val/epoch_999/"


for protocal in A B D
do
    echo "**protocal" $protocal"**"
    for retrieval_type in T2M M2T
    do
        echo $retrieval_type
        python retrieval.py --retrieval_type $retrieval_type --protocal $protocal --expdirs $path1 $path2 $path3 
    done
done
