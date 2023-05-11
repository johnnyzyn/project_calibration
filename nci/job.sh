#!/bin/bash

#PBS -l ncpus=12
#PBS -l mem=30GB
#PBS -l jobfs=200GB
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -P li96
#PBS -l walltime=4:00:00
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l wd

module load python3/3.9.2
module load pytorch/1.9.0
cd /scratch/li96/lt2442/project_calibration/
WANDB_MODE=offline python3 train_other_loss.py --archi_num ${ARCH_NUM} --loss ${LOSS} --tss_simple_dir ./nats_bench//NATS-sss-v1_0-50262-simple