#!/bin/bash
#SBATCH -J batch_process_data   # TODO: job name, you can change it
#SBATCH --time=00-03:00:00  # TODO: requested time 3 hour. If you don't set it, only 15min
#SBATCH -p gpu    # running on "batch" partition/queue
#SBATCH --gres=gpu:1  # TODO: select your GPU
#SBATCH -N 1   # 1 nodes
#SBATCH	-n 1   # 8 cpu cores
#SBATCH -c 8   # 1 cpu core per task
#SBATCH --mem=16g  # TODO: requesting 16GB of RAM total or more
#SBATCH --output=logs/test.%j.out  #saving standard output to file, can change the name from 'myjob' to anything
#SBATCH --error=logs/test.%j.err   #saving standard error to file, can change the name from 'myjob' to anything

module use /cluster/tufts/nerf2robotics/modules
module load conda-env/nerf2-py3.13.5

python batch_process_raw.py -v ../../../Pilot2_MIMO/F1_MIMO_train_raw ../../../Pilot2_MIMO/F1_MIMO_train_processed
# python batch_process_raw.py -v ../../../Pilot2_MIMO/F2_MIMO_train_raw ../../../Pilot2_MIMO/F2_MIMO_train_processed
