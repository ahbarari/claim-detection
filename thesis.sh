#!/bin/bash

# SBATCH --ntasks=1
# SBATCH --partition=A40short
# SBATCH --gres=gpu:4
# SBATCH --time=08:00:00
# SBATCH --nodes=1

module load Miniforge3
# conda init
source ~/.bashrc
conda_env="thesisenv"
conda activate $conda_env
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s6mamirh/.conda/envs/env/lib
python /home/s6ambara/thesis-codebase/roberta_test.py
# python /home/s6ambara/thesis-codebase/compute_metrics.py

conda deactivate
