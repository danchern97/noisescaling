#!/bin/bash

N_AGENTS=5
COUNT_PER_AGENT=4

for i in $(seq 1 $N_AGENTS); do 
    sbatch run_agent.job $COUNT_PER_AGENT
done