#!/bin/bash 
#SBATCH --job-name=pt_guides
#SBATCH --time=24:00:00
#SBATCH -p ellenyeh
#SBATCH -N 1
#SBATCH -n 32 
source ~/.bash_profile

conda activate env/

genes=resources/gene.txt
safe=resources/safe_region.final.bed
sat=resources/saturating_gene.txt

python design_guides.py -i $genes -t 32 -o results_final --num_guides 7 --safe_region_bed $safe --saturating_genes $sat

#python design_guides.py -i essential.txt -p essential --num_guides 20
#python design_guides.py -i non_essential.txt -p non_essential --num_guides 16
