#!/bin/bash 
#SBATCH --job-name=pt_guides
#SBATCH --time=2:00:00
#SBATCH -p normal,ellenyeh,owners
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G

source ~/.bash_profile

conda activate env/
ml ncbi-blast+

python design_guides_mp.py -t 64 -o results_mp -i gene.txt -p test_output_mp --num_guides 8
