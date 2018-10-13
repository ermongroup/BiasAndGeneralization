#!/usr/bin/env bash

#python gan.py --gpu=0 --dataset=count_1 &
#python gan.py --gpu=1 --dataset=count_2 &
#python gan.py --gpu=2 --dataset=count_3 &
#python gan.py --gpu=3 --dataset=count_1_2 &
#python gan.py --gpu=4 --dataset=count_2_3 &
#python gan.py --gpu=5 --dataset=count_1_3 &
#python gan.py --gpu=6 --dataset=count_1_2_3 &
#python gan.py --gpu=7 --dataset=count_4 &


#python gan.py --gpu=0 --dataset=disjoint_count_1 --model=gaussian_wgan &
#python gan.py --gpu=1 --dataset=disjoint_count_2 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_3 --model=gaussian_wgan &
#python gan.py --gpu=3 --dataset=disjoint_count_4 --model=gaussian_wgan  &
#python gan.py --gpu=4 --dataset=disjoint_count_5 --model=gaussian_wgan  &
#python gan.py --gpu=5 --dataset=disjoint_count_6 --model=gaussian_wgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_7 --model=gaussian_wgan  &
#python gan.py --gpu=7 --dataset=disjoint_count_8 --model=gaussian_wgan  &
#python gan.py --gpu=0 --dataset=disjoint_count_9 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=disjoint_count_10 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_11 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=disjoint_count_12 --model=gaussian_wgan  &

#python gan.py --gpu=0 --dataset=disjoint_count_3_10 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=disjoint_count_4_9 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_5_8 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=disjoint_count_6_7 --model=gaussian_wgan  &
#python gan.py --gpu=4 --dataset=disjoint_count_3_10 --model=gaussian_dcgan  &
#python gan.py --gpu=5 --dataset=disjoint_count_4_9 --model=gaussian_dcgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_5_8 --model=gaussian_dcgan  &
#python gan.py --gpu=7 --dataset=disjoint_count_6_7 --model=gaussian_dcgan  &

#python gan.py --gpu=0 --dataset=disjoint_count_3_6_9 --model=gaussian_wgan  &
#python gan.py --gpu=1 --dataset=disjoint_count_4_6_8 --model=gaussian_wgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_3_6_9_10 --model=gaussian_wgan  &
#python gan.py --gpu=3 --dataset=disjoint_count_3_4_5_6_7_8 --model=gaussian_wgan  &
#
#python gan.py --gpu=4 --dataset=disjoint_count_3_6_9 --model=gaussian_dcgan  &
#python gan.py --gpu=5 --dataset=disjoint_count_4_6_8 --model=gaussian_dcgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_3_6_9_10 --model=gaussian_dcgan  &
#python gan.py --gpu=7 --dataset=disjoint_count_3_4_5_6_7_8 --model=gaussian_dcgan  &

#python gan.py --gpu=4 --dataset=disjoint_count_1 --model=gaussian_dcgan &
#python gan.py --gpu=5 --dataset=disjoint_count_2 --model=gaussian_dcgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_3 --model=gaussian_dcgan &
#python gan.py --gpu=7 --dataset=disjoint_count_4 --model=gaussian_dcgan  &
#python gan.py --gpu=4 --dataset=disjoint_count_5 --model=gaussian_dcgan  &
#python gan.py --gpu=5 --dataset=disjoint_count_6 --model=gaussian_dcgan  &
#python gan.py --gpu=6 --dataset=disjoint_count_7 --model=gaussian_dcgan  &
#python gan.py --gpu=7 --dataset=disjoint_count_8 --model=gaussian_dcgan  &
#python gan.py --gpu=0 --dataset=disjoint_count_9 --model=gaussian_dcgan  &
#python gan.py --gpu=1 --dataset=disjoint_count_10 --model=gaussian_dcgan  &
#python gan.py --gpu=2 --dataset=disjoint_count_11 --model=gaussian_dcgan  &
#python gan.py --gpu=3 --dataset=disjoint_count_12 --model=gaussian_dcgan  &

#python gan.py --gpu=0 --dataset=color_gap_2 --model=gaussian_wgan &
#python gan.py --gpu=1 --dataset=color_gap_5 --model=gaussian_wgan &
#python gan.py --gpu=2 --dataset=color_gap_10 --model=gaussian_wgan &
#python gan.py --gpu=3 --dataset=color_gap_20 --model=gaussian_wgan &
#python gan.py --gpu=0 --dataset=color_gap_40 --model=gaussian_wgan &
#python gan.py --gpu=1 --dataset=color_gap_45 --model=gaussian_wgan &
#python gan.py --gpu=4 --dataset=color_gap_2 --model=gaussian_dcgan &
#python gan.py --gpu=5 --dataset=color_gap_5 --model=gaussian_dcgan &
#python gan.py --gpu=6 --dataset=color_gap_10 --model=gaussian_dcgan &
#python gan.py --gpu=7 --dataset=color_gap_20 --model=gaussian_dcgan &
#python gan.py --gpu=2 --dataset=color_gap_40 --model=gaussian_dcgan &
#python gan.py --gpu=3 --dataset=color_gap_45 --model=gaussian_dcgan &
#python vae.py --gpu=4 --dataset=color_gap_2 --beta=50.0 &
#python vae.py --gpu=5 --dataset=color_gap_5 --beta=50.0 &
#python vae.py --gpu=6 --dataset=color_gap_10 --beta=50.0 &
#python vae.py --gpu=7 --dataset=color_gap_20 --beta=50.0 &
#python vae.py --gpu=6 --dataset=color_gap_40 --beta=50.0 &
#python vae.py --gpu=7 --dataset=color_gap_45 --beta=50.0 &

#
python vae.py --gpu=0 --z_dim=10 --dataset=disjoint_count_1 --beta=$1 --run=$2 &
python vae.py --gpu=1 --z_dim=10 --dataset=disjoint_count_2 --beta=$1 --run=$2 &
python vae.py --gpu=2 --z_dim=10 --dataset=disjoint_count_3 --beta=$1 --run=$2 &
python vae.py --gpu=3 --z_dim=10 --dataset=disjoint_count_4 --beta=$1 --run=$2 &
python vae.py --gpu=4 --z_dim=12 --dataset=disjoint_count_5 --beta=$1 --run=$2 &
python vae.py --gpu=5 --z_dim=14 --dataset=disjoint_count_6 --beta=$1 --run=$2 &
python vae.py --gpu=6 --z_dim=16 --dataset=disjoint_count_7 --beta=$1 --run=$2 &
python vae.py --gpu=7 --z_dim=18 --dataset=disjoint_count_8 --beta=$1 --run=$2 &
python vae.py --gpu=4 --z_dim=20 --dataset=disjoint_count_9 --beta=$1 --run=$2 &
python vae.py --gpu=5 --z_dim=22 --dataset=disjoint_count_10 --beta=$1 --run=$2 &
python vae.py --gpu=6 --z_dim=24 --dataset=disjoint_count_11 --beta=$1 --run=$2 &
python vae.py --gpu=7 --z_dim=26 --dataset=disjoint_count_12 --beta=$1 --run=$2 &