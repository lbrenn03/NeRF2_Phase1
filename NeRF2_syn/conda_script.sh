#!/bin/bash
#SBATCH -J cmccal04_nerf2_test
#SBATCH --time=00-02:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16g
#SBATCH --output=logs/nerf2_test.%j.out
#SBATCH --error=logs/nerf2_test.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cmccal04@tufts.edu

module use /cluster/tufts/nerf2robotics/modules
module load conda-env/nerf2-py3.13.5

cd /cluster/tufts/nerf2robotics/cmccal04/NeRF2  # TODO: change to your NeRF2 path

python nerf2_runner_plot.py --mode test --config configs/ble-rssi.yml --dataset_type ble --gpu 0
