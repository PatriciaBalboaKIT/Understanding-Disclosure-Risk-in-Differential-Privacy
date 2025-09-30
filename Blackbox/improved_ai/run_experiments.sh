#!/bin/bash

echo "Starting script"
if [[ $# -eq 0 ]] ; then # if called with no arguments
    echo "Usage: bash $0 <DATASET NAME>"
    exit 1
fi

DATASET=$1
CODE=main.py

if [ $DATASET == 'census' ]; then
    L2_RATIO=1e-6
    SKEW_OUT=3
    SENS_OUT=3
    SENS_ATTR=1
else
    L2_RATIO=1e-3
    SKEW_OUT=0
    SENS_OUT=0
    SENS_ATTR=2
fi

echo "Filling data/ directory"
python $CODE $DATASET --use_cpu=0 --save_data=1 --skew_attribute=0 --skew_outcome=$SKEW_OUT --sensitive_outcome=$SENS_OUT --target_test_train_ratio=0.5 --target_data_size=50000

echo "Beginning experiment"
python $CODE $DATASET \
--use_cpu=0 \
--skew_attribute=0 \
--skip_corr=1 \
--skew_outcome=$SKEW_OUT \
--sensitive_outcome=$SENS_OUT \
--target_test_train_ratio=0.5 \
--target_data_size=50000 \
--candidate_size=10000 \
--target_model='nn' \
--target_epochs=30 \
--target_l2_ratio=$L2_RATIO \
--target_learning_rate=0.001 \
--target_batch_size=500 \
--target_clipping_threshold=4 \
--target_privacy='grad_pert' \
--attribute=$SENS_ATTR \
--target_epsilon=1 \
--run=0