#!/bin/bash 
#SBATCH --job-name=pt_guides
#SBATCH --time=8:00:00
#SBATCH -p normal,ellenyeh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G

source ~/.bash_profile

conda activate env/
ml ncbi-blast+

python design_guides.py -i gene.txt -t 64 -p test_output --num_guides 8
