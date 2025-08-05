#!/bin/bash

PROJECT="noisescaling"
ENTITY="m-d-cherniavskii-university-of-amsterdam"
SWEEP_ID="sxis9iv6"

N_AGENTS=3
COUNT_PER_AGENT=5

source .env

for i in $(seq 0 $((N_AGENTS - 1))); do 
    echo "Starting agent $i"
    CUDA_VISIBLE_DEVICES=$i wandb agent -p $PROJECT -e $ENTITY --count $COUNT_PER_AGENT $SWEEP_ID > logs/agent_$i.log 2>&1 &
    PID=$!
    echo "Agent $i PID: $PID"
    echo $PID >> agent_pids.txt
done