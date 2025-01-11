#!/bin/bash 
#SBATCH --job-name=pt_guides_parallel
#SBATCH --time=48:00:00
#SBATCH -p ellenyeh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=1G

source ~/.bash_profile

conda activate env/
ml ncbi-blast+

gene_file=resources/gene.txt
prefix=guides
outdir=results

mkdir -p $outdir

split_dir=$outdir/split

if [ -d $split_dir ]; then
	
	rm -r $split_dir 
fi

mkdir $outdir/split
split -l 1000 $gene_file $outdir/split/$gene_file_

slurmids="" # storage of slurm job ids
for input_file in $outdir/split/$gene_file_*; do
	sbatch --wait -p ellenyeh,owners,normal -c 15 --mem=4G -t 8:00:00 --job-name pt_guide_design --wrap="python design_guides.py -i $input_file -t 16 -o $outdir -p $prefix --num_guides 8" &
done

sleep 2

echo "waiting for jobs to finish"
wait
echo "all jobs complete"

rm $outdir/$prefix.csv
rm $outdir/$prefix.summary.csv

cat $outdir/work/*/*.filtered.csv >> $outdir/$prefix.filtered.csv
cat $outdir/work/*/*.summary.csv >> $outdir/$prefix.summary.csv
