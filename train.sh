#!/bin/bash
#SBATCH --clusters=amarel
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu005,gpu006,gpu007,gpu008,gpu009,gpu010,gpu011,gpu012,gpu013,gpu014,gpu015,gpu016,gpu017,gpu018,gpu019,gpu020
#SBATCH --job-name=thermompnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm.%N.%j.log
#SBATCH --error=slurm.%N.%j.err
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --begin=now
#SBATCH --open-mode=append

ulimit -n 65535
python thermompnn_train.py config.yaml $@

