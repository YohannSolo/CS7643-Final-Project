#!/bin/bash
#SBATCH -J Mistral7B
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH --mem-per-gpu=30G
#SBATCH -t720
#SBATCH -oReport-8.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wlai46@gatech.edu
cd $SLURM_SUBMIT_DIR

module load anaconda3
srun python r16.py