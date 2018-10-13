#!/usr/bin/env bash

## Experiments for single mode generalization -- size
#python train.py --gpu=0 --dataset=pie_30_0_3 --objective=$1 --run=$2 & 
#python train.py --gpu=1 --dataset=pie_30_0_4 --objective=$1 --run=$2 &
#python train.py --gpu=2 --dataset=pie_30_0_5 --objective=$1 --run=$2 &
#python train.py --gpu=3 --dataset=pie_30_0_6 --objective=$1 --run=$2 &
#python train.py --gpu=4 --dataset=pie_30_0_7 --objective=$1 --run=$2 &
#
##
### Experiments for single mode generalization -- location
#python train.py --gpu=5 --dataset=pie_30_1_3 --objective=$1 --run=$2 &
#python train.py --gpu=6 --dataset=pie_30_1_4 --objective=$1 --run=$2 &
#python train.py --gpu=7 --dataset=pie_30_1_5 --objective=$1 --run=$2 &
#python train.py --gpu=0 --dataset=pie_30_1_6 --objective=$1 --run=$2 &
#python train.py --gpu=1 --dataset=pie_30_1_7 --objective=$1 --run=$2 &
#
##
### Experiments for single mode generalization -- color
#python train.py --gpu=2 --dataset=pie_30_3_1 --objective=$1 --run=$2 &
#python train.py --gpu=3 --dataset=pie_30_3_3 --objective=$1 --run=$2 &
#python train.py --gpu=4 --dataset=pie_30_3_4 --objective=$1 --run=$2 &
#python train.py --gpu=5 --dataset=pie_30_3_5 --objective=$1 --run=$2 &
#python train.py --gpu=6 --dataset=pie_30_3_6 --objective=$1 --run=$2 &
#python train.py --gpu=7 --dataset=pie_30_3_7 --objective=$1 --run=$2 &
#python train.py --gpu=0 --dataset=pie_30_3_8 --objective=$1 --run=$2 &
#python train.py --gpu=1 --dataset=pie_30_3_9 --objective=$1 --run=$2 &
#
## Experiments for prototypical enhancement
#python train.py --gpu=2 --dataset=pie_30_3_34 --objective=$1 --run=$2 &
#python train.py --gpu=3 --dataset=pie_30_3_35 --objective=$1 --run=$2 &
#python train.py --gpu=4 --dataset=pie_30_3_36 --objective=$1 --run=$2 &
#python train.py --gpu=5 --dataset=pie_30_3_37 --objective=$1 --run=$2 &
#python train.py --gpu=6 --dataset=pie_30_3_38 --objective=$1 --run=$2 &
#python train.py --gpu=7 --dataset=pie_30_3_39 --objective=$1 --run=$2 &

# Experiments for stability
#python train.py --gpu=0 --dataset=pie_3_3_349 --objective=$1 --run=$2 &
#python train.py --gpu=1 --dataset=pie_6_3_349 --objective=$1 --run=$2 &
#python train.py --gpu=2 --dataset=pie_9_3_349 --objective=$1 --run=$2 &
#python train.py --gpu=3 --dataset=pie_18_3_349 --objective=$1 --run=$2 &
#python train.py --gpu=4 --dataset=pie_30_3_349 --objective=$1 --run=$2 &
#python train.py --gpu=5 --dataset=pie_60_3_349 --objective=$1 --run=$2 &
#python train.py --gpu=6 --dataset=pie_90_3_349 --objective=$1 --run=$2 &
#python train.py --gpu=7 --dataset=pie_150_3_349 --objective=$1 --run=$2 &
##
#python train.py --gpu=0 --dataset=pie_3_1_459 --objective=$1 --run=$2 &
#python train.py --gpu=1 --dataset=pie_6_1_459 --objective=$1 --run=$2 &
#python train.py --gpu=2 --dataset=pie_9_1_459 --objective=$1 --run=$2 &
#python train.py --gpu=3 --dataset=pie_18_1_459 --objective=$1 --run=$2 &
#python train.py --gpu=4 --dataset=pie_30_1_459 --objective=$1 --run=$2 &
#python train.py --gpu=5 --dataset=pie_60_1_459 --objective=$1 --run=$2 &
#python train.py --gpu=6 --dataset=pie_90_1_459 --objective=$1 --run=$2 &
#python train.py --gpu=7 --dataset=pie_150_1_459 --objective=$1 --run=$2 &
#
#python train.py --gpu=0 --dataset=pie_3_0_349 --objective=$1 --run=$2 &
#python train.py --gpu=1 --dataset=pie_6_0_349 --objective=$1 --run=$2 &
#python train.py --gpu=2 --dataset=pie_9_0_349 --objective=$1 --run=$2 &
#python train.py --gpu=3 --dataset=pie_18_0_349 --objective=$1 --run=$2 &
#python train.py --gpu=4 --dataset=pie_30_0_349 --objective=$1 --run=$2 &
#python train.py --gpu=5 --dataset=pie_60_0_349 --objective=$1 --run=$2 &
#python train.py --gpu=6 --dataset=pie_90_0_349 --objective=$1 --run=$2 &
#python train.py --gpu=7 --dataset=pie_150_0_349 --objective=$1 --run=$2 &

## Experiments for random memorization
python train.py --gpu=0 --dataset=pie_10 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=1 --dataset=pie_15 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=2 --dataset=pie_20 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=3 --dataset=pie_30 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=4 --dataset=pie_50 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=5 --dataset=pie_60 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=6 --dataset=pie_75 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=7 --dataset=pie_100 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=0 --dataset=pie_150 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=1 --dataset=pie_200 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=2 --dataset=pie_300 --objective=$1 --run=$2 --architecture=$3 &
python train.py --gpu=3 --dataset=pie_400 --objective=$1 --run=$2 --architecture=$3 &

#python train.py --gpu=0 --dataset=pie_20 --objective=gan --run=0 --architecture=small &
#python train.py --gpu=1 --dataset=pie_50 --objective=gan --run=1 --architecture=small &

#python train.py --gpu=0 --dataset=pie_10 --objective=$1 --run=4 &
#python train.py --gpu=1 --dataset=pie_15 --objective=$1 --run=4 &
#python train.py --gpu=2 --dataset=pie_20 --objective=$1 --run=4 &
#python train.py --gpu=3 --dataset=pie_30 --objective=$1 --run=4 &
#python train.py --gpu=4 --dataset=pie_50 --objective=$1 --run=4 &
#python train.py --gpu=5 --dataset=pie_60 --objective=$1 --run=4 &
#python train.py --gpu=6 --dataset=pie_75 --objective=$1 --run=4 &
#python train.py --gpu=7 --dataset=pie_100 --objective=$1 --run=4 &
#python train.py --gpu=4 --dataset=pie_150 --objective=$1 --run=4 &
#python train.py --gpu=5 --dataset=pie_200 --objective=$1 --run=4 &
#python train.py --gpu=6 --dataset=pie_300 --objective=$1 --run=4 &
#python train.py --gpu=7 --dataset=pie_400 --objective=$1 --run=4 &