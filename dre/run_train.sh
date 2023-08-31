#!/bin/bash

# ERM model
python3 train.py --dataset terra_incognita --model ERM

# DRE model
python3 train.py --dataset terra_incognita --model DRE