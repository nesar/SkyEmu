#!/bin/sh
#SBATCH -p cp100

echo [$SECONDS] setting up environment
#export KERAS_BACKEND=tensorflow

srun -p cp100 python run_latentdim.py

echo [$SECONDS] End job 


