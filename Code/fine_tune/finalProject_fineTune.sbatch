#!/bin/bash
#SBATCH -J Mistral7B_fine_tune
#SBATCH -N1 --gres=gpu:H100:2 --ntasks-per-node=4
#SBATCH --mem-per-gpu=30G
#SBATCH -t480
#SBATCH -ofine_tune_-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ckniffin6@gatech.edu
cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate hf
srun python fine_tune.py
