#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=VocabEmbeddings
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:50:00
#SBATCH --output=ConceptSimWordnet_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

# Your job starts in the directory where you call sbatch
cd $HOME/Discover-then-Name
# Activate your environment
source activate fact_env

srun python get_closest_concepts_wordnet.py