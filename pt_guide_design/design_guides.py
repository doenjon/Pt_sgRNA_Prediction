# design sgRNA guides for Phaeodactylum tricornutum
# Doenier 2022

import os
import argparse
from Bio import SeqIO
import gffutils
import pybedtools
from pybedtools import BedTool
import subprocess
import pandas as pd
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pyfaidx import Fasta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("guide_design.log")
    ]
)

class GuideDesigner:
    def __init__(self, args):
        with open(args.i, "r") as infile:
            self.target_genes = infile.read().strip().split("\n")

        if not os.path.exists(args.o):
            os.makedirs(args.o)

        self.outdir = args.o
        self.output_file_base = os.path.join(args.o, args.p)
        self.num_guides = args.num_guides
        self.variants = args.variants
        self.genome = args.genome
        self.chrom_sizes = args.chrom_sizes
        self.gff_file = args.gff
        self.gff = self.load_gff(args.gff)
        self.bed = args.bed
        self.args = args

    def run(self):
        """Design knockout guides for target genes"""
        ko_guides_df, summary_df = self.design_knockout_guides()
        print(f"Designed {len(ko_guides_df)} guides for {len(self.target_genes)} genes")
        
        # Save results
        ko_guides_df.to_csv(f"{self.output_file_base}.KO.csv", index=False, sep="\t")
        summary_df.to_csv(f"{self.output_file_base}.KO.summary.csv", index=False, sep="\t")
        self.make_bed(ko_guides_df).saveas(f"{self.output_file_base}.KO.bed")

    def design_knockout_guides(self):
        """Design guides for all target genes using parallel processing"""
        result_df = pd.DataFrame()
        summary_df = pd.DataFrame()

        with ThreadPoolExecutor(max_workers=self.args.t) as executor:
            futures = {executor.submit(self._design_knockout_guides, gene): gene 
                      for gene in self.target_genes}

            with tqdm(total=len(futures), desc="Designing Guides") as pbar:
                for future in as_completed(futures):
                    try:
                        guides, summary = future.result()
                        result_df = pd.concat([result_df, guides], axis=0)
                        summary_df = pd.concat([summary_df, summary], axis=0)
                    except Exception as e:
                        logging.error(f"Error for target gene {futures[future]}: {e}")
                        traceback.print_exc()
                    pbar.update(1)

        return result_df, summary_df

    def _design_knockout_guides(self, gene):
        """Design guides for a single gene"""
        logging.info(f"Designing guides for {gene}")

        outdir = os.path.join(self.args.o, "work", gene)
        os.makedirs(outdir, exist_ok=True)

        # Get gene sequence
        seq_fn, seq = self.get_gene_sequence(gene, flank=50, outdir=outdir)

        # Design guides using CRISPOR
        guide_df = self.crispor(gene, seq_fn, outdir, flank=50)
        
        # Annotate guide locations
        pos_df = self.annotate_guide_loc(guide_df["targetSeq"], outdir=outdir)
        guide_df = guide_df.merge(pos_df, on="targetSeq", how="left")

        # Get CRISPRon scores
        crispron_df = self.crispron(seq_fn, outdir)
        guide_df = guide_df.merge(crispron_df, on="targetSeq", how="left")

        # Filter and score guides
        guide_df, filter_summary = self.filter_guides(gene, guide_df)

        # Save intermediate results
        guide_df.to_csv(os.path.join(outdir, f"{gene}.filtered.csv"), 
                       index=False, sep="\t")
        
        return guide_df, pd.DataFrame([filter_summary])

    def filter_guides(self, gene, df):
        """Filter guides based on various quality criteria"""
        logging.debug("Filtering guides")

        # Add cutsite and exon annotations
        df['cutsite'] = list(map(self.get_cutsite, df["start"], df["stop"], df["strand"]))
        df['in_exon'] = list(map(self.region_in_exon, df["seqId"], df['chrom'], 
                                df['start'], df['stop'], df['strand'], 
                                [self.gff]*len(df)))

        # Calculate exon penalty
        exon_penalty = 0.25
        df['num_exons_in_gene'] = self.get_num_exon_in_gene(gene, self.gff)
        df["exon_penalty"] = np.where((df["num_exons_in_gene"] == 1) | 
                                     (df["in_exon"] < df["num_exons_in_gene"]), 
                                     0, exon_penalty)

        # Get relative position in gene
        df['relative_pos'] = list(map(self.get_relative_position_in_gene, 
                                     [gene]*len(df), df["cutsite"], 
                                     [self.gff]*len(df)))

        # Calculate off-target penalties
        mismatch_seed_penalty = 0.25
        mismatch_2_penalty = 0.25
        mismatch_3_penalty = 0.05
        mismatch_4_penalty = 0.025
        
        df["offtarget_penalty"] = (mismatch_seed_penalty * df["mismatch_seed"] + 
                                  mismatch_2_penalty * df["mismatch_2"] + 
                                  mismatch_3_penalty * df["mismatch_3"] + 
                                  mismatch_4_penalty * df["mismatch_4"])

        # Check for variants
        df = self.annotate_variants(df)

        # Calculate final design score
        pos_weight = 0.5
        crispron_weight = 1/60
        df["design_score"] = (crispron_weight * df["CRISPRon"] - 
                             pos_weight * (1.5*df["relative_pos"])**4 - 
                             df["exon_penalty"] - 
                             df["offtarget_penalty"])

        # Apply filters
        pos_filter = (df.relative_pos <= 0.9)
        mismatch_0_filter = (df.mismatch_0 == 0)
        mismatch_1_filter = (df.mismatch_1 == 0)
        mismatch_seed_filter = (df.mismatch_seed <= 20)
        mismatch_2_filter = (df.mismatch_2 <= 20)
        graf_filter = (df.GrafEtAlStatus == "GrafOK")
        variant_filter = (df.has_variant == False)
        exon_filter = (df.in_exon > 0)
        blast_filter = (df.start.notna())

        filters = (pos_filter & mismatch_0_filter & mismatch_1_filter & 
                  mismatch_seed_filter & mismatch_2_filter & graf_filter & 
                  variant_filter & exon_filter & blast_filter)

        # Generate filter summary
        filter_summary = {
            "gene": gene,
            "guides_pre_filter": len(df),
            "PP:pos<0.9": pos_filter.sum() / len(df),
            "PP:mismatch_0": mismatch_0_filter.sum() / len(df),
            "PP:mismatch_1": mismatch_1_filter.sum() / len(df),
            "PP:mismatch_seed<20": mismatch_seed_filter.sum() / len(df),
            "PP:mismatch_2<20": mismatch_2_filter.sum() / len(df),
            "PP:graf": graf_filter.sum() / len(df),
            "PP:no_variant": variant_filter.sum() / len(df),
            "PP:in_valid_exon": exon_filter.sum() / len(df),
            "PP:blast_filter": blast_filter.sum() / len(df),
            "PP:all": filters.sum() / len(df)
        }

        # Apply filters and sort
        df = df[filters].copy()
        df = df.sort_values(by="design_score", ascending=False)

        # Handle overlapping guides
        df = self.find_overlapping(df)
        df = df[df["overlapping"] == False]
        
        # Take top guides
        df = df.head(self.num_guides)
        filter_summary["guides_post_filter"] = len(df)

        if len(df) < self.num_guides:
            logging.warning(f"Only designed {len(df)}/{self.num_guides} guides for {gene}")

        return df, filter_summary

    def find_overlapping(self, df, overlap_threshold=0.2):
        """Find overlapping guides and mark them"""
        df["overlapping"] = False
        df = df.reset_index(drop=True)

        for guide1, _ in df.iterrows():
            for guide2 in range(0, guide1):
                if not df["overlapping"][guide2]:
                    stop = min(df["stop"][guide1], df["stop"][guide2])
                    start = max(df["start"][guide1], df["start"][guide2])
                    overlap = (stop - start) / 23  # Assuming guide length of 23

                    if overlap > 0 and overlap >= overlap_threshold:
                        df.loc[guide1, "overlapping"] = True
                        break

        return df

    def get_cutsite(self, start, end, strand):
        """Calculate cutsite position based on guide location"""
        if strand == "+":
            return end - 6
        elif strand == "-":
            return start + 6
        return np.nan

    def get_relative_position_in_gene(self, target_gene, cutsite, gff):
        """Calculate relative position of cutsite in gene"""
        if pd.isna(cutsite):
            return np.nan

        gene_name = f"gene:{target_gene}"
        gene = gff[gene_name]
        
        exon_len = 0
        exon_len_to_cut = 0
        
        exons = list(gff.children(gene, featuretype='exon'))
        exons = sorted(exons, key=lambda x: int(x.attributes["rank"][0]))

        for exon in exons:
            exon_len += exon.stop - exon.start

        if gene.strand == "+":
            for exon in exons:
                if cutsite >= exon.stop:
                    exon_len_to_cut += exon.stop - exon.start
                elif exon.start <= cutsite <= exon.stop:
                    exon_len_to_cut += cutsite - exon.start

        if gene.strand == "-":
            for exon in exons:
                if cutsite <= exon.start:
                    exon_len_to_cut += exon.stop - exon.start
                elif exon.start <= cutsite <= exon.stop:
                    exon_len_to_cut += exon.stop - cutsite
        
        return exon_len_to_cut / exon_len

    # Helper methods remain largely unchanged...
    # [Previous helper methods for load_gff, crispor, crispron, etc.]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Design CRISPR guides for gene targeting",
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

    parser.add_argument('-t',
                        help='number cores to use',
                        type=int,
                        default=4,
                        required=False)

    parser.add_argument('--num_guides',
                        help='number of guides to design for each target',
                        default=7,
                        type=int,
                        required=False)

    parser.add_argument('--genome',
                        help='fasta of genome to use for guide design',
                        type=str,
                        required=True)

    parser.add_argument('--gff',
                        help='gff3 for genome to use for guide design',
                        type=str,
                        required=True)
    
    parser.add_argument('--bed',
                        help='bed for genome to use for guide design',
                        type=str,
                        required=True)
    
    parser.add_argument('--variants',
                        help='variants to account for in guide design',
                        type=str,
                        required=True)
    
    parser.add_argument('--chrom_sizes',
                        help='chrom_sizes for bedtools',
                        type=str,
                        required=True)

    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    designer = GuideDesigner(args)
    designer.run()




