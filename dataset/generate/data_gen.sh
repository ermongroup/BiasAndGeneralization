#!/usr/bin/env bash

python2 dots_generator.py --bn=0 --bs=8192 --count=$1 &
python2 dots_generator.py --bn=1 --bs=8192 --count=$1 &
python2 dots_generator.py --bn=2 --bs=8192 --count=$1 &
python2 dots_generator.py --bn=3 --bs=8192 --count=$1 &
python2 dots_generator.py --bn=4 --bs=8192 --count=$1 &
python2 dots_generator.py --bn=5 --bs=8192 --count=$1 &
python2 dots_generator.py --bn=6 --bs=8192 --count=$1 &
python2 dots_generator.py --bn=7 --bs=8192 --count=$1 &

