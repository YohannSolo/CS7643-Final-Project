#!/bin/bash
#SBATCH -J Mistral7B_16_v2
#SBATCH -N1 --gres=gpu:H100:2 --ntasks-per-node=8
#SBATCH --mem-per-gpu=40G
#SBATCH -t480
#SBATCH -olora_16_v2-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ckniffin6@gatech.edu
cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate hugging_face
srun python lora_main_16.py
