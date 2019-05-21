#!/bin/bash
#SBATCH --account=punim0811
#SBATCH --job-name=test
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=richard.wang1@unimelb.edu.au
#SBATCH --mail-type=FAIL,END
module load Python/3.6.4-intel-2017.u2-GCC-6.2.0-CUDA10
module load Tensorflow/1.13.1-intel-2017.u2-GCC-6.2.0-CUDA10-Python-3.6.4-GPU
python job_dataset_config.py > job_dataset_config.txt