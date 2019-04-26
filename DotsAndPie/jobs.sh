#!/usr/bin/env bash

#python train.py --gpu=1 --dataset=dots_1 --log_path=/data/BiasAndGeneralization/dots/ --run=$1 &
#python train.py --gpu=2 --dataset=dots_2 --log_path=/data/BiasAndGeneralization/dots/ --run=$1 &
#python train.py --gpu=3 --dataset=dots_3 --log_path=/data/BiasAndGeneralization/dots/ --run=$1 &
#python train.py --gpu=1 --dataset=dots_4 --log_path=/data/BiasAndGeneralization/dots/ --run=$1 &
#python train.py --gpu=2 --dataset=dots_5 --log_path=/data/BiasAndGeneralization/dots/ --run=$1 &
#python train.py --gpu=3 --dataset=dots_6 --log_path=/data/BiasAndGeneralization/dots/ --run=$1 &
#python train.py --gpu=1 --dataset=dots_7 --log_path=/data/BiasAndGeneralization/dots/ --run=$1 &
#python train.py --gpu=2 --dataset=dots_8 --log_path=/data/BiasAndGeneralization/dots/ --run=$1 &
#python train.py --gpu=3 --dataset=dots_9 --log_path=/data/BiasAndGeneralization/dots/ --run=$1 &

python train.py --gpu=1 --dataset=pie_150_0_59 --log_path=/data/BiasAndGeneralization/pie/ --run=$1 &
python train.py --gpu=2 --dataset=pie_150_0_58 --log_path=/data/BiasAndGeneralization/pie/ --run=$1 &
python train.py --gpu=3 --dataset=pie_150_0_57 --log_path=/data/BiasAndGeneralization/pie/ --run=$1 &
python train.py --gpu=1 --dataset=pie_150_0_56 --log_path=/data/BiasAndGeneralization/pie/ --run=$1 &
python train.py --gpu=2 --dataset=pie_150_0_5  --log_path=/data/BiasAndGeneralization/pie/ --run=$1 &
python train.py --gpu=3 --dataset=pie_150_0_54 --log_path=/data/BiasAndGeneralization/pie/ --run=$1 &
python train.py --gpu=1 --dataset=pie_150_0_53 --log_path=/data/BiasAndGeneralization/pie/ --run=$1 &
python train.py --gpu=2 --dataset=pie_150_0_52 --log_path=/data/BiasAndGeneralization/pie/ --run=$1 &
python train.py --gpu=3 --dataset=pie_150_0_51 --log_path=/data/BiasAndGeneralization/pie/ --run=$1 &



