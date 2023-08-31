#!/bin/bash

# DomainBed IRM
python3 -m domainbed.scripts.train \
       --data_dir=./data/ \
       --algorithm IRM \
       --dataset terra_incognita \
       --test_env 0

# DomainBed GroupDRO
python3 -m domainbed.scripts.train \
       --data_dir=./data/ \
       --algorithm GroupDRO \
       --dataset terra_incognita \
       --test_env 0

# DomainBed Mixup
python3 -m domainbed.scripts.train \
       --data_dir=./data/ \
       --algorithm Mixup \
       --dataset terra_incognita \
       --test_env 0