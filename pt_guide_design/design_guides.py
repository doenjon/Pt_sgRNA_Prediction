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
import re
import sys

# Add the parent directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from sgrna_scorer.predict import SgRNAScorer
except ImportError as e:
    logging.error(f"Could not import sgrna_scorer from {project_root}. Error: {e}")
    raise

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
        # Read input sequence from FASTA file
        with open(args.i, "r") as infile:
            lines = infile.readlines()
            # Skip header line(s) and concatenate sequence lines
            sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
            self.sequence = ''.join(sequence_lines)

        if not os.path.exists(args.o):
            os.makedirs(args.o)

        self.outdir = args.o
        self.output_file_base = os.path.join(args.o, args.p)
        self.num_guides = args.num_guides
        self.genome = args.genome
        self.args = args

        # Create genome index if it doesn't exist
        if not all(os.path.exists(f"{self.genome}{ext}") for ext in [".amb", ".ann", ".bwt", ".pac", ".sa"]):
            logging.info("Creating BWA index for genome...")
            cmd = f"bwa index {self.genome}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to create BWA index: {result.stderr}")
                raise RuntimeError("Failed to create BWA index for genome")

        # Initialize sgRNA scorer
        try:
            self.scorer = SgRNAScorer()
            logging.info("Successfully initialized sgRNA scorer")
        except Exception as e:
            logging.error(f"Failed to initialize sgRNA scorer: {e}")
            raise

    def run(self):
        """Design guides for input sequence"""
        guides_df, summary_df = self.design_guides()
        print(f"Designed {len(guides_df)} guides for input sequence")
        
        # Save results
        guides_df.to_csv(f"{self.output_file_base}.guides.csv", index=False, sep="\t")
        summary_df.to_csv(f"{self.output_file_base}.summary.csv", index=False, sep="\t")

    def design_guides(self):
        """Design guides for input sequence"""
        # Write sequence to temp file for CRISPOR
        seq_fn = os.path.join(self.outdir, "input_sequence.fa")
        with open(seq_fn, "w") as f:
            f.write(f">input_sequence\n{self.sequence}\n")

        # Design guides using CRISPOR
        guide_df = self.crispor("input_sequence", seq_fn, self.outdir)
        
        # Filter and score guides
        guide_df, filter_summary = self.filter_guides(guide_df)

        return guide_df, pd.DataFrame([filter_summary])

    def filter_guides(self, df):
        """Filter guides based on quality criteria"""
        logging.debug("Filtering guides")

        # Score guides using sgrna-scorer
        try:
            # Extract 20nt guide sequences (excluding PAM)
            sequences = [seq[:-3] for seq in df["targetSeq"].tolist()]  # Remove last 3 nucleotides (PAM)
            scores = self.scorer.predict_sequences(sequences)
            df["sgRNA_Scorer"] = scores  # Changed column name
            logging.info(f"Successfully scored {len(sequences)} guide sequences")
        except Exception as e:
            logging.error(f"Failed to score guides: {e}")
            df["sgRNA_Scorer"] = 0.5  # Default score if scoring fails
        
        # Calculate off-target penalties
        mismatch_seed_penalty = 0.25
        mismatch_2_penalty = 0.25
        mismatch_3_penalty = 0.05
        mismatch_4_penalty = 0.025

        df["offtarget_penalty"] = (mismatch_seed_penalty * df["mismatch_seed"] + 
                                  mismatch_2_penalty * df["mismatch_2"] + 
                                  mismatch_3_penalty * df["mismatch_3"] + 
                                  mismatch_4_penalty * df["mismatch_4"])

        # Design score based only on off-targets
        df["design_score"] = -df["offtarget_penalty"]  # Removed activity score from design score

        # Apply filters
        mismatch_0_filter = (df.mismatch_0 == 0)
        mismatch_1_filter = (df.mismatch_1 == 0)
        mismatch_seed_filter = (df.mismatch_seed <= 20)
        mismatch_2_filter = (df.mismatch_2 <= 20)
        graf_filter = (df.GrafEtAlStatus == "GrafOK")

        filters = (mismatch_0_filter & mismatch_1_filter & 
                  mismatch_seed_filter & mismatch_2_filter & graf_filter)

        # Generate filter summary
        filter_summary = {
            "guides_pre_filter": len(df),
            "PP:mismatch_0": mismatch_0_filter.sum() / len(df),
            "PP:mismatch_1": mismatch_1_filter.sum() / len(df),
            "PP:mismatch_seed<20": mismatch_seed_filter.sum() / len(df),
            "PP:mismatch_2<20": mismatch_2_filter.sum() / len(df),
            "PP:graf": graf_filter.sum() / len(df),
            "PP:all": filters.sum() / len(df)
        }

        # Apply filters and sort
        df = df[filters].copy()
        df = df.sort_values(by="design_score", ascending=False)
        
        # Take top guides
        df = df.head(self.num_guides)
        filter_summary["guides_post_filter"] = len(df)

        if len(df) < self.num_guides:
            logging.warning(f"Only designed {len(df)}/{self.num_guides} guides")
        
        return df, filter_summary

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

    def crispor(self, name, seq_fn, outdir, flank=0):
        """Run CRISPOR to find potential guides and their off-targets"""
        outfile_crispor = f"{outdir}/{name}_guides.csv"
        outfile_offtarget = f"{outdir}/{name}_offtarget.csv"

        # Get absolute path to CRISPOR script
        crispor_path = os.path.join(os.path.dirname(__file__), "crisporWebsite", "crispor.py")
        resources_dir = os.path.join(os.path.dirname(__file__), "resources")

        # Don't rerun expensive command if not necessary
        if not os.path.exists(outfile_crispor):
            cmd = f"python {crispor_path} --noEffScores --genomeDir {resources_dir} --offtargets {outfile_offtarget} ens79PhaTri {seq_fn} {outfile_crispor}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"CRISPOR failed with error: {result.stderr}")
                raise RuntimeError(f"CRISPOR command failed: {cmd}")
            
            if not os.path.exists(outfile_crispor):
                logging.error("CRISPOR did not generate output file")
                raise RuntimeError(f"CRISPOR failed to generate output file: {outfile_crispor}")

        guide_df = pd.read_csv(outfile_crispor, sep='\t', header=0)
        guide_df.rename({'#seqId': 'seqId'}, axis=1, inplace=True)
        guide_df["pos"] = guide_df['guideId'].map(lambda p: int(re.sub('\D', '', p)) - flank)

        offtarget_df = pd.read_csv(outfile_offtarget, sep="\t", header=0)
        offtarget_df = self.count_offtargets(offtarget_df)
        
        guide_df = guide_df.merge(offtarget_df, left_on="targetSeq", right_on="guideSeq", how="left")
        guide_df[offtarget_df.columns] = guide_df[offtarget_df.columns].fillna(0)
        
        return guide_df

    def count_offtargets(self, offtarget_df):
        """Count different types of off-targets for each guide"""
        # Group by the two columns and calculate duplicate counts
        duplicate_counts = offtarget_df.groupby(['guideSeq', 'mismatchCount']).size().reset_index(name='duplicate_count')

        # Pivot the duplicate_counts DataFrame
        pivot_table = duplicate_counts.pivot_table(index='guideSeq', 
                                                 columns='mismatchCount', 
                                                 values='duplicate_count', 
                                                 fill_value=0)

        pivot_table.columns = [f'mismatch_{col}' for col in pivot_table.columns]
        
        for col in ['mismatch_0', 'mismatch_1', 'mismatch_2', 'mismatch_3', 'mismatch_4']:
            if col not in pivot_table.columns:
                pivot_table[col] = 0

        # Count seed mismatches
        offtarget_df['seed_mismatch'] = offtarget_df['mismatchPos'].str[-8:].str.count('\*')

        duplicate_counts = offtarget_df.groupby(['guideSeq', 'seed_mismatch']).size().reset_index(name='duplicate_count')
        seed_pivot = duplicate_counts.pivot_table(index='guideSeq', 
                                                columns='seed_mismatch', 
                                                values='duplicate_count', 
                                                fill_value=0)
        
        seed_pivot.columns = [f'seed_mismatch_{col}' for col in seed_pivot.columns]
        
        for col in ['seed_mismatch_0', 'seed_mismatch_1', 'seed_mismatch_2', 
                    'seed_mismatch_3', 'seed_mismatch_4']:
            if col not in seed_pivot.columns:
                seed_pivot[col] = 0

        seed_pivot["mismatch_seed"] = seed_pivot["seed_mismatch_0"]        
        
        # Merge all off-target information
        offtarget_df = offtarget_df.merge(pivot_table, left_on='guideSeq', right_index=True, how='left')
        offtarget_df = offtarget_df.merge(seed_pivot, left_on='guideSeq', right_index=True, how='left')
        offtarget_df = offtarget_df.drop_duplicates(subset=['guideSeq'])

        return offtarget_df[["guideSeq", "mismatch_seed", "mismatch_0", "mismatch_1", 
                            "mismatch_2", "mismatch_3", "mismatch_4"]]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Design CRISPR guides for input sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i',
                        help='File containing input sequence',
                        type=str,
                        required=True)

    parser.add_argument('-p',
                        help='output prefix',
                        type=str,
                        default="guides",
                        required=False)

    parser.add_argument('-o',
                        help='output_dir',
                        type=str,
                        default="results",
                        required=False)

    parser.add_argument('-t',
                        help='number cores to use',
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument('--num_guides',
                        help='number of guides to design',
                        default=7,
                        type=int,
                        required=False)

    parser.add_argument('--genome',
                        help='fasta of genome to use for off-target search',
                        default="resources/Phaeodactylum_tricornutum.ASM15095v2.dna.toplevel.fa",
                        type=str,
                        required=False)

    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    designer = GuideDesigner(args)
    designer.run()




