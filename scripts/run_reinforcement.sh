#!/bin/bash

export CONF_FILE=../conf/config_ppo_contrastive_m2

python ../src/reinforcement_training_block.py $CONF_FILE
