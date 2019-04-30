#!/bin/bash
#SBATCH --job-name=gan
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=weiw8@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
module load Python/3.5.2-intel-2017.u2
module load Tensorflow/1.4.0-intel-2017.u2-Python-3.5.2-gpu
module load CUDA/8.0.44-intel-2017.u2
module load cuDNN/6.0-intel-2017.u2-CUDA-8.0.44
python job1/sngan_hinge.py
