# design sgRNA guides for Phaeodactylum tricornutum
# Doenier 2022

import os 
import argparse
from Bio import SeqIO
import gffutils
from gffutils.helpers import asinterval
from gffutils import FeatureDB
import pybedtools
import subprocess
import pandas as pd
import re
import tempfile
import math
import time
import sys
from dna_features_viewer import BiopythonTranslator
from BCBio import GFF
import glob 

class Guide_Designer():

	def __init__(self, args):

		with open(args.i, "r") as infile:
			self.target_genes = infile.read().strip().split("\n")

			
		if not os.path.exists(args.o):
			os.makedirs(args.o)
		self.output_file = os.path.join(args.o, args.p)
		self.num_guides = args.num_guides
		self.genome = args.genome
		self.annotation = args.annotation
		self.chrom_sizes = args.chrom_sizes
		self.gff = self.load_gff(self.annotation) 

	def design_guides(self):

		res = pd.DataFrame()

		for gene in self.target_genes:
			gene_guides = self.crispor(gene)
			res = res.append(gene_guides)
		
		res.to_csv(f"{self.output_file}.csv", index=False, sep="\t", header=True)
		
		self.make_bed(res)

	def load_gff(self, gff_file, force_rebuild=False):
		''' load a gff file or gff database '''

		db_file = "resources/pt_genome_gff.sql"

		if force_rebuild or not os.path.exists(db_file):
			gffutils.create_db(gff_file, db_file, merge_strategy="create_unique", force=True)

		return gffutils.FeatureDB(db_file)

	def get_gene_sequence(self, gene, flank=50):
		''' Get the sequence of a gene from a gene identifier '''

		# Very confused about this generator bs. definitely a cleaner way to do this...
		# also, this feels very convoluted...

		gene_name = f"gene:{gene}" # gff is weird?

		gene_interval = [asinterval(self.gff[gene_name])]
		gene_gtf = pybedtools.BedTool((g for g in gene_interval))
		extended_genes = gene_gtf.slop(s=True, b=flank, g=self.chrom_sizes)

		tmp_gene = f"results/tmp/{gene}.fa"
		extended_genes.sequence(fi=self.genome, fo=tmp_gene, s=True, name=True)

		seq = open(extended_genes.seqfn).read().strip().split("\n")[1]
		with open(extended_genes.seqfn, "w") as file:
			file.write(f">{gene}\n")
			file.write(f"{seq}\n")

		return extended_genes.seqfn

	def region_in_exon(self, gene, chrom, start, stop, strand):

		cutsite = self.get_cutsite(start, stop, strand)

		genes = [g for g in self.gff.region(seqid=chrom, start=min(start, stop), end=max(start, stop), completely_within=False, featuretype="gene")]
		
		if len(genes) != 1:	# guide must be in exactly one gene
			return False 
	
		curr_gene = genes[0]["ID"][0].split(":")[1]	# gene must be targeted gene
		if curr_gene != gene: 
			return False

		exons = [e for e in self.gff.region(seqid=chrom, start=min(start, stop), end=max(start, stop), completely_within=False, featuretype="exon")]

		if len(exons) != 1:	# guide must be in exactly one exon
			return False 

		curr_exon = exons[0]

		cutsite_buffer = 2
		if (cutsite - cutsite_buffer >= curr_exon.start) and (cutsite + cutsite_buffer <= curr_exon.stop):	# guide must be completely in exon (excessive...)
			return True

		return False

	def get_cutsite(self, start, end, strand):

		if strand == "+":
			return end - 6
		elif strand == "-":
			return start + 6
		return None
		
	def crispor(self, gene):
		''' gene - String name of gene '''

		print(f"Generating guides for {gene}")

		flank = 50
		seq_fn = self.get_gene_sequence(gene, flank=flank)
		outfile = f"results/guides/{gene}.csv"
		if not os.path.exists(outfile):
			cmd = f"/home/groups/ellenyeh/jdoenier/mageck/wang_subsample/crisporWebsite/env_p2/bin/python crisporWebsite/crispor.py ens79PhaTri {seq_fn} {outfile}"
			subprocess.run(cmd, shell=True, capture_output=True)
		else:
			print("\tusing already computed guides")

		#read guide result
		df = pd.read_csv(outfile, sep='\t', header=0)
		df.rename({'#seqId': 'seqId'}, axis=1, inplace=True)

		#Write the position to the df
		df["pos"] = df['guideId'].map(lambda p: int(re.sub('\D', '', p)))

		df = self.blast_guides(df)

		# determine if guide in target gene exon
		df['in_exon'] = list(map(self.region_in_exon, df["seqId"], df['chrom'], df['start'], df['stop'], df['strand']))
		#filters - only keep guides in target exon, GrafOK, good specificity scores
		df = df[df.in_exon == True]
		df = df[df.GrafEtAlStatus == "GrafOK"]
		df = df[df.mitSpecScore != "None"]
		df = df[df.cfdSpecScore != "None"]

		df["mitSpecScore"] = df['mitSpecScore'].map(lambda i: int(i))
		df["cfdSpecScore"] = df['cfdSpecScore'].map(lambda i: int(i))

		# only keep high specificity score
		df = df[df.mitSpecScore > 50]
		df = df[df.cfdSpecScore > 50]

		df = df.sort_values("pos")

		# pick all guides within first 300 bp of start, increase threshold till you get enough guides
		pos = 300
		filtered_df = df[df.pos < pos]
		while filtered_df.shape[0] < self.num_guides and filtered_df.shape[0] < df.shape[0]:
			pos += 50
			filtered_df = df[df.pos < pos]
			
			# got to end of gene
			if pos > df["pos"].max(axis=0):
				break

		filtered_df = filtered_df.sort_values(by=["offtargetCount", "mitSpecScore", "cfdSpecScore", "Doench '16-Score"], ascending=[True, False, False, False]).head(self.num_guides)
		
		print(f"\tPicked {filtered_df.shape[0]} guides")
		return filtered_df

	def blast_guides(self, df):
		""" df must have guideId and targetseq """
		df_fasta = df[["guideId", "targetSeq"]].copy()
		df_fasta["guideId"] = df_fasta['guideId'].map(lambda s: f">{s}")

		tmp_fasta = f"results/tmp/{time.time_ns()}.input"
		df_fasta.to_csv(tmp_fasta, index=False, sep="\n", header=0)

		subprocess.run(f"ml ncbi-blast+", shell=True, capture_output=True)

		tmp_blast = f"results/tmp/{time.time_ns()}.blast"

		cmd = f"blastn -query {tmp_fasta} -db resources/Phaeodactylum_tricornutum.ASM15095v2.dna.toplevel.fa -task blastn-short -evalue 0.0000035 -outfmt 6 > {tmp_blast}"
		subprocess.run(cmd, shell=True, capture_output=True)

		tmp_bed = f"results/tmp/{time.time_ns()}.bed"

		cmd = f"blast2bed/blast2bed {tmp_blast}; mv {tmp_blast}.bed {tmp_bed}"
		subprocess.run(cmd, shell=True, capture_output=True)

		df_bed = pd.read_csv(tmp_bed, sep='\t', header=0, names=["chrom", "start", "stop", "guideId", "x", "strand"])
		
		merged_df = pd.merge(df, df_bed, on="guideId")
		return merged_df

	def make_bed(self, df):

		df_fasta = df[["seqId", "targetSeq"]].copy()
		df_fasta["seqId"] = df_fasta['seqId'].map(lambda s: f">{s}")

		tmp_fasta = f"results/tmp/{time.time_ns()}"
		df_fasta.to_csv(tmp_fasta, index=False, sep="\n", header=0)

		subprocess.run(f"ml ncbi-blast+", shell=True, capture_output=True)
		tmp_blast = f"results/tmp/{time.time_ns()}"

		cmd = f"blastn -query {tmp_fasta} -db resources/Phaeodactylum_tricornutum.ASM15095v2.dna.toplevel.fa -task blastn-short -evalue 0.0000035 -outfmt 6 > {tmp_blast}"
		subprocess.run(cmd, shell=True, capture_output=True)

		bed = f"{self.output_file}.bed"

		cmd = f"blast2bed/blast2bed {tmp_blast}; mv {tmp_blast}.bed {bed}"
		subprocess.run(cmd, shell=True, capture_output=True)

	def draw_guides(self):
		""" broken """
		for gene in self.target_genes:

			gene = self.gff[f"gene:{gene}"]
			gene_region = [g for g in self.gff.region(region=(gene.seqid, gene.start, gene.end), completely_within=True)]	

			tmp_gff = f"results/tmp/{time.time_ns()}.gff"
			with open(tmp_gff, "w") as gff_file:
				for f in gene_region:
					gff_file.write(str(f) + '\n')

			# design = dpl.load_design_from_gff("plasmid.gff", "chrom1", region=[0, 6451])
			
			in_handle = open(tmp_gff)
			for rec in GFF.parse(in_handle):
			    print(rec.features[0])
			in_handle.close()


			graphic_record = BiopythonTranslator().translate_record(record)
			graphic_record.plot(ax=ax1, with_ruler=False, strand_in_label_threshold=4)
			# record = BiopythonTranslator().translate_record(tmp_gff)
			# for r in record.features:
			# 	print(r)			# record = record.crop((gene.start, gene.end))
			# ax, _ = record.plot(figure_width=10, strand_in_label_threshold=7)
			# # ax = BiopythonTranslator.quick_class_plot(tmp_gff, figure_width=9)
			# # ax.figure.savefig('dfv.jpg', bbox_inches='tight')
			# # ax, _ = graphic_record.plot(figure_width=10, strand_in_label_threshold=7)
			# ax.figure.savefig('sequence_and_translation.png', bbox_inches='tight')

	def off_target_validation(self, csv):

		# convert to fasta
		tmp_fasta = f"results/tmp/{time.time_ns()}"
		with open(csv, "r") as in_file:
			with open(tmp_fasta, "w") as out_file:
				for line in infile:
					seqId, seq = line.strip().split(",")
					tmp_fasta.write(f">{seqId}\n{seq}\n")

		subprocess.run(f"ml ncbi-blast+", shell=True, capture_output=True)

		cmd = f"blastn -query {tmp_fasta} -db resources/Phaeodactylum_tricornutum.ASM15095v2.dna.toplevel.fa -task blastn-short -outfmt 6 > {tmp_blast}"
		subprocess.run(cmd, shell=True, capture_output=True)

	def clean_tmp(self):
		files = glob.glob("results/tmp/*")
		for f in files:
			os.remove(f)

if __name__ == "__main__":

	# PARAMETERS
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)

	

	parser.add_argument('-i',
						help='File of genes to generate guides for',
						type=str,
						required=True)

	parser.add_argument('-p',
						help='output prefix',
						type=str,
						default="pt_guides",
						required=False)

	parser.add_argument('-o',
						help='output_dir',
						type=str,
						default="results",
						required=False)

	parser.add_argument('--num_guides',
						help='number of guides to design for each target',
						default=10,
						type=int,
						required=False)

	parser.add_argument('--genome',
						help='fasta of genome to use for guide design',
						default="resources/Phaeodactylum_tricornutum.ASM15095v2.dna.toplevel.fa",
						type=str,
						required=False)

	parser.add_argument('--annotation',
						help='gff for genome to use for guide design',
						default="resources/Phaeodactylum_tricornutum.ASM15095v2.52.gff3",
						type=str,
						required=False)
	parser.add_argument('--chrom_sizes',
						help='chrom_sizes for bedtools',
						default="resources/Phaeodactylum_tricornutum.ASM15095v2.dna.chrom.sizes",
						type=str,
						required=False)

	args = parser.parse_args()

	designer = Guide_Designer(args)
	designer.design_guides()
	# designer.clean_tmp()
	# designer.draw_guides()


	"""def build_gene_sequences(self, force_rebuild=False):
		''' build a file of genomic gene sequences including flanking regions '''

		gene_w_flank_file = "resources/pt_genes_w_flank.fasta"
		
		if force_rebuild or not os.path.exists(gene_w_flank_file):


			def features():
				for gene in self.gff.features_of_type("gene"):
					yield asinterval(gene)
			# gene_ids = []
			# for gene in self.gff.features_of_type("gene"):
			# 	gene_ids.append(gene.id.split(":")[1])

			gene_gtf = pybedtools.BedTool(features()).saveas('resources/genes.gtf')
			# gene_gtf = pybedtools.BedTool((gene for gene in self.gff.features_of_type('gene'))).saveas('resources/genes.gtf')
			extended_genes = gene_gtf.slop(s=True, b=50, g=self.chrom_sizes)
			extended_genes.sequence(fi=self.genome, fo=gene_w_flank_file, s=True, name=True)"""