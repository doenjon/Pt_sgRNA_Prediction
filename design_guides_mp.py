# design sgRNA guides for Phaeodactylum tricornutum
# Doenier 2022

import os 
import argparse
from Bio import SeqIO
import gffutils
from gffutils.helpers import asinterval
from gffutils import FeatureDB
import pybedtools
from pybedtools import BedTool
from pybedtools.cbedtools import create_interval_from_list
import subprocess
import pandas as pd
import re
import tempfile
import math
import time
import sys
from tqdm import tqdm
import numpy as np
from dna_features_viewer import BiopythonTranslator
from sklearn.preprocessing import MinMaxScaler
from BCBio import GFF
import glob 
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import traceback
import cProfile


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)


logging.basicConfig(
    level=logging.WARN,  # Set the minimum level for log messages
    # level=logging.INFO,  # Set the minimum level for log messages
    # level=logging.DEBUG,  # Set the minimum level for log messages
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Guide_Designer():

    def __init__(self, args):

        with open(args.i, "r") as infile:
            self.target_genes = infile.read().strip().split("\n")

        if not os.path.exists(args.o):
            os.makedirs(args.o)

        self.output_file = os.path.join(args.o, args.p)
        self.num_guides = args.num_guides
        self.variants = args.variants
        self.genome = args.genome
        self.chrom_sizes = args.chrom_sizes
        self.gff_file = args.gff
        # self.gff = self.load_gff(args.gff) 
        self.bed = args.bed 
        self.args = args # fuck me lazy



    def process_gene(self, target_gene):
            try:
                guides, summary = self.design_guides(target_gene)

                return guides, summary

            except Exception as e:
                logging.error(f"Error for target gene {target_gene}: {e}")
                traceback.print_exc(file=sys.stdout)
                return None, None

    def run(self):

        logging.info(f"Designing {self.num_guides} guides/gene for {len(self.target_genes)} genes")

        result_df = pd.DataFrame()
        summary_df = pd.DataFrame()

        # total_guides = 0
        # subpar_genes = []

        # for gene in self.target_genes:
        #     res, summy = self.design_guides(gene)
        #     result_df = pd.concat([result_df, res], axis=0)
        #     summary_df = pd.concat([summary_df, summy], axis=0)
    


        with ProcessPoolExecutor(max_workers=self.args.t) as executor:

            futures = {executor.submit(self.process_gene, target_gene): target_gene for target_gene in self.target_genes}
            total_tasks = len(futures)
            completed_tasks = 0

            with tqdm(total=total_tasks, desc="Designing Guides") as pbar:
                for future in as_completed(futures):
                    completed_tasks += 1
                    pbar.update(1)

                    guides, summary = future.result()

                    if guides is not None and summary is not None:
                        result_df = pd.concat([result_df, guides], axis=0)
                        summary_df = pd.concat([summary_df, summary], axis=0)


        result_df.to_csv(f"{self.output_file}.csv", index=False, sep="\t", header=True)
        summary_df.to_csv(f"{self.output_file}.summary.csv", index=False, sep="\t", header=True)


    def design_guides(self, gene):
        ''' gene - String name of gene '''

        os.makedirs(os.path.join(self.args.o, "guides"), exist_ok=True)
        outfile = os.path.join(self.args.o, "guides", f"{gene}.csv")
        summary_file = os.path.join(self.args.o, "guides", f"{gene}.summary")
        
        if os.path.exists(outfile):
            logging.info(f"guides already calculated for {gene}. Will not recalculate")

            guide_df = pd.read_csv(outfile, header=0, sep="\t")
            summary_df = pd.read_csv(summary_file, header=0, sep="\t")

        else:
            logging.info(f"Designing guides from scratch for {gene}")

            outdir = os.path.join(self.args.o, "tmp", f"{gene}")
            os.makedirs(outdir, exist_ok=True)

            summary_data = {}
            summary_data["gene"] = gene

            gff = self.load_gff(self.gff_file) 

            flank = 50
            seq_fn, seq = self.get_gene_sequence(gene, flank=flank, outdir=outdir)

            guide_df = self.crispor(gene, seq_fn, outdir, flank)
            
            summary_data["guides possible"] = len(guide_df)

            # annotate chrom, start, stop
            guide_df = self.blast_guides(gene, gff, guide_df, outdir)

            crispron_df = self.crispron(gene, seq_fn, outdir)

            guide_df = guide_df.merge(crispron_df, on="targetSeq", how="left")

            all_guides_file = os.path.join(outdir, f"{gene}.all.csv")

            guide_df.to_csv(all_guides_file, sep="\t", header=True, index=False)

            guide_df, filter_summary = self.filter_guides(gene, guide_df, gff)

            summary_data.update(filter_summary)
            summary_data["guides designed"] = len(guide_df)
            summary_data["average position"] = guide_df["relative_pos"].mean()
            summary_data["average CRISPROn"] = guide_df["CRISPRon"].mean()
            summary_data["average design_score"] = guide_df["design_score"].mean()
            summary_data["average mismatch_3"] = guide_df["mismatch_3"].mean()
            summary_data["average mismatch_4"] = guide_df["mismatch_4"].mean()

            column_order = ["guides designed", "average position", "average CRISPROn", "average design_score", \
                            "average mismatch_3", "average mismatch_4", "PP:mismatch_0", "PP:mismatch_1", "PP:mismatch_2", \
                            "PP:mismatch_seed", "PP:graf", "PP:no_variant", "PP:in_exon", "PP:blast_filter", "PP:all"]
            
            # print(summary_data)

            summary_df = pd.DataFrame(summary_data, columns=column_order, index=[0])

            guide_df.to_csv(f"{outfile}", index=False, header=True, sep="\t")
            summary_df.to_csv(f"{summary_file}", index=False, header=True, sep="\t")

            logging.info(summary_df)

        return guide_df, summary_df


    def filter_guides(self, gene, df, gff):
        # df of guides

        logging.debug("Filtering guides")
        #read guide result
        # small hacky
        outfile = os.path.join(self.args.o, "guides", f"{gene}.csv")


        # add some more annotations

        df['cutsite'] = list(map(self.get_cutsite, df["start"], df["stop"], df["strand"]))
        df['in_exon'] = list(map(self.region_in_exon, df["seqId"], df['chrom'], df['start'], df['stop'], df['strand'], [gff]*len(df)))

        # don't pick guides in the last exon unless you have to... bad NMD (presumably)
        exon_penalty = 0.25
        df['num_exons_in_gene'] = self.get_num_exon_in_gene(gene, gff)
        df["exon_penalty"] = np.where((df["num_exons_in_gene"] == 1) | (df["in_exon"] < df["num_exons_in_gene"]), 0, exon_penalty)

        # Annotate cut position in gene
        df['relative_pos'] = list(map(self.get_relative_position_in_gene, [gene for _ in range(len(df))], df["cutsite"], [gff]*len(df)))
      
        # off-target penalty
        mismatch_3_penalty = 0.2
        mismatch_4_penalty = 0.05
        df["offtarget_penalty"] = mismatch_3_penalty*df["mismatch_3"] + mismatch_4_penalty*df["mismatch_4"]

        df = self.annotate_variants(df)

        # # overlap penalty
        # df['overlap_penalty'] = 0

        # # Iterate through each row
        # for index, row in df.iterrows():
        #     overlap_penalty = 0.1
        #     for _, other_row in df.iterrows():
        #         if not pd.isna(other_row) and not pd.isna(row):
        #             if (
        #                 other_row['targetSeq'] != row['targetSeq']
        #                 and other_row['start'] <= row['stop']
        #                 and other_row['stop'] >= row['start']
        #             ):
        #                 if other_row['CRISPRon_norm'] > row['CRISPRon_norm']:
        #                     df.at[index, 'overlap_penalty'] = overlap_penalty


        df["design_score"] = df["CRISPRon_norm"] - df["relative_pos"] - df["exon_penalty"] - df["offtarget_penalty"] #- df["overlap_penalty"]

        outfile = os.path.join(self.args.o, "tmp", f"{gene}.all.score.csv")
        df.to_csv(outfile, header=True, index=False, sep="\t")

        # Filter guides for deleterious qualities

        mismatch_0_filter    = ( df.mismatch_0 == 0 )
        mismatch_1_filter    = ( df.mismatch_1 == 0 )
        mismatch_2_filter    = ( df.mismatch_2 == 0 )
        mismatch_seed_filter = ( df.offtargets_w_perfect_12_seed == 0 )
        graf_filter          = ( df.GrafEtAlStatus == "GrafOK" )
        variant_filter       = ( df.has_variant == False )
        exon_filter          = ( df.in_exon == True )
        blast_filter         = ( df.start.notna())

        filters = mismatch_0_filter & mismatch_1_filter & mismatch_2_filter & mismatch_seed_filter & graf_filter & variant_filter & exon_filter & blast_filter

        filter_summary = {}
        filter_summary["PP:mismatch_0"]    = mismatch_0_filter.sum()
        filter_summary["PP:mismatch_1"]    = mismatch_1_filter.sum()
        filter_summary["PP:mismatch_2"]    = mismatch_2_filter.sum()
        filter_summary["PP:mismatch_seed"] = mismatch_seed_filter.sum()
        filter_summary["PP:graf"]          = graf_filter.sum()
        filter_summary["PP:no_variant"]    = variant_filter.sum()
        filter_summary["PP:in_exon"]       = exon_filter.sum()
        filter_summary["PP:blast_filter"]  = blast_filter.sum()
        filter_summary["PP:all"]           = filters.sum()

        num_guides = len(df)
        filter_summary = {key: '{:.1%}'.format(value / num_guides) for key, value in filter_summary.items()}

        df = df[filters]

        df = df.sort_values(by="design_score", ascending=False)

        df = df.head(self.num_guides)

        if len(df) < self.num_guides:
            logging.warning(f"Failed to designed required number of guides for {gene}. Designed {len(df)} / {self.num_guides}")

        # logging.info(f"\tPicked {df.shape[0]} guides for gene {gene}")
        
        return df, filter_summary

    
    def load_gff(self, gff_file, force_rebuild=False):
        ''' load a gff file or gff database '''

        db_file = "resources/pt_genome_gff.sql"

        if force_rebuild or not os.path.exists(db_file):
            print("Creating database from gff")
            gffutils.create_db(gff_file, db_file, merge_strategy="create_unique", force=True)

        db = gffutils.FeatureDB(db_file)

        # db = gffutils.create_db(gff_file, ":memory:", merge_strategy="create_unique", force=True)

        return db 

        
      # print("Calculating introns... this may take awhile")

      # introns = db.create_introns()
      # print("updating db")
      # db.update(introns, merge_strategy="create_unique") # calculate introns
      # print("done")
        

        return db


    def crispor(self, gene, seq_fn, outdir, flank):

        outfile_crispor = f"{outdir}/{gene}_guides.csv"
        outfile_offtarget = f"{outdir}/{gene}_offtarget.csv"

        cmd = f"/home/groups/ellenyeh/jdoenier/mageck/wang_subsample/crisporWebsite/env_p2/bin/python crisporWebsite/crispor.py --offtargets {outfile_offtarget} ens79PhaTri {seq_fn} {outfile_crispor}"
        subprocess.run(cmd, shell=True, capture_output=True)

        guide_df = pd.read_csv(outfile_crispor, sep='\t', header=0)
        guide_df.rename({'#seqId': 'seqId'}, axis=1, inplace=True)
        guide_df["pos"] = guide_df['guideId'].map(lambda p: int(re.sub('\D', '', p)) - flank)

        offtarget_df = pd.read_csv(outfile_offtarget, sep="\t", header=0)

        offtarget_df = self.count_offtargets(offtarget_df)
        
        guide_df = guide_df.merge(offtarget_df, left_on="targetSeq", right_on="guideSeq", how="left")

        return guide_df

    
    def count_offtargets(self, offtarget_df):

        # Group by the two columns and calculate duplicate counts
        duplicate_counts = offtarget_df.groupby(['guideSeq', 'mismatchCount']).size().reset_index(name='duplicate_count')

        # Pivot the duplicate_counts DataFrame
        pivot_table = duplicate_counts.pivot_table(index='guideSeq', columns='mismatchCount', values='duplicate_count', fill_value=0)

        pivot_table.columns = [f'mismatch_{col}' for col in pivot_table.columns]
        
        for col in ['mismatch_0', 'mismatch_1', 'mismatch_2', 'mismatch_3', 'mismatch_4']:
            if col not in pivot_table.columns:
                pivot_table[col] = 0

        # Merge the pivot table into the original DataFrame
        offtarget_df = offtarget_df.merge(pivot_table, left_on='guideSeq', right_index=True, how='left')


        # Now repeat for seed mismatches
        offtarget_df['seed_mismatch'] = offtarget_df['mismatchPos'].str[-12:].str.count('\*')

        duplicate_counts = offtarget_df.groupby(['guideSeq', 'seed_mismatch']).size().reset_index(name='duplicate_count')
        pivot_table = duplicate_counts.pivot_table(index='guideSeq', columns='seed_mismatch', values='duplicate_count', fill_value=0)
        # only care about perfect seed mathes

        pivot_table.columns = [f'seed_mismatch_{col}' for col in pivot_table.columns]
        
        for col in ['seed_mismatch_0', 'seed_mismatch_1', 'seed_mismatch_2', 'seed_mismatch_3', 'seed_mismatch_4']:
            if col not in pivot_table.columns:
                pivot_table[col] = 0

        pivot_table["offtargets_w_perfect_12_seed"] = pivot_table["seed_mismatch_0"]        
        offtarget_df = offtarget_df.merge(pivot_table, left_on='guideSeq', right_index=True, how='left')


        offtarget_df = offtarget_df.drop_duplicates(subset=['guideSeq'])
        # offtarget_df.rename(columns={0: 'mismatch_0', 1: 'mismatch_1', 2: 'mismatch_2', 3: 'mismatch_3', 4: 'mismatch_4'}, inplace=True)

        offtarget_df = offtarget_df[["guideSeq", "offtargets_w_perfect_12_seed", "mismatch_0", "mismatch_1", "mismatch_2", "mismatch_3", "mismatch_4"]]

        print(offtarget_df)
        return offtarget_df


    def crispron(self, gene, seq_fn, outdir):

        outdir_score = os.path.join(self.args.o, "tmp", f"{gene}_crispron")
        cmd = f"crispron/bin/CRISPRon.sh {seq_fn} {outdir}"
        subprocess.run(cmd, shell=True, capture_output=True)
        
        crispron_df  = pd.read_csv(f"{outdir}/crispron.csv")

        crispron_df["targetSeq"]  = crispron_df["30mer"].str[4:-3]
        
        crispron_df["CRISPRon_norm"] = MinMaxScaler().fit_transform(crispron_df[['CRISPRon']])

        crispron_df = crispron_df[["targetSeq", "CRISPRon", "CRISPRon_norm"]]

        return crispron_df


    def get_gene_sequence(self, target_gene, flank, outdir):

        # Load the gene BED file and the reference FASTA file using pybedtools
        genes = BedTool(self.bed)
        fasta = BedTool(self.genome)

        # Find the gene's interval, then slop on flank size
        gene_interval = genes.filter(lambda feature: feature.name == target_gene).slop(s=True, b=flank, g=self.chrom_sizes)

        sequence_file = gene_interval.sequence(fi=fasta, s=True, name=True).seqfn
        sequence = open(sequence_file).read().strip().split("\n")[1]
        
        tmp_gene_file = os.path.join(outdir, f"{target_gene}.fa")

        with open(tmp_gene_file, "w") as file:
            file.write(f">{target_gene}\n")
            file.write(sequence)
        
        return tmp_gene_file, sequence


    def get_num_exon_in_gene(self, target_gene, gff):
        
        gene_name = f"gene:{target_gene}" # gff is weird?
        gene = gff[gene_name]
        
        return len(list(gff.children(gene, featuretype='exon')))
    

    def get_relative_position_in_gene(self, target_gene, cutsite, gff):
        # return [0, 1] position of cut site in gene coding sequence
        
        gene_name = f"gene:{target_gene}" # gff is weird?
        gene = gff[gene_name]
        
        if pd.isna(cutsite):
            return np.nan

        exon_len = 0
        exon_len_to_cut = 0
        
        for exon in gff.children(gene, featuretype='exon'):
            exon_len += exon.stop - exon.start

            if cutsite >= exon.stop:
                exon_len_to_cut += exon.stop - exon.start
                
            # cutsite is in this exon
            elif exon.start <= cutsite <= exon.stop:
                if gene.strand == "+":
                    exon_len_to_cut += cutsite - exon.start
                elif gene.strand == "-":
                    exon_len_to_cut += exon.stop - cutsite

        
        return exon_len_to_cut / exon_len


    def region_in_exon(self, gene, chrom, start, stop, strand, gff):

        # Guide was not properly identified in genome - redundant to check all of them...

        # print(f"region_in_exon: {gene} @ {chrom}:{start}-{stop} ({strand})")

        # if np.isnan(start) or np.isnan(stop) or np.isnan(strand):
        #     return False 

        logging.debug(f"region_in_exon: {gene} @ {chrom}:{start}-{stop} ({strand})")

        if pd.isna(start) or pd.isna(stop) or pd.isna(strand):
            return False 

        cutsite = self.get_cutsite(start, stop, strand)

        genes = [g for g in gff.region(seqid=chrom, start=min(start, stop), end=max(start, stop), completely_within=False, featuretype="gene")]
        
        if len(genes) != 1: # guide must be in exactly one gene
            # print(f"WARNING - region_in_exon() - guide at cutsite {cutsite}: found more than 1 gene")
            return False 
    
        curr_gene = genes[0]["ID"][0].split(":")[1] # gene must be targeted gene
        if curr_gene != gene: 
            # print(f"WARNING - region_in_exon() - guide at cutsite {cutsite}: found wrong gene")
            return False

        exons = [e for e in gff.region(seqid=chrom, start=min(start, stop), end=max(start, stop), completely_within=False, featuretype="exon")]

        if len(exons) == 0: # guide must be in exactly one exon
            # print(f"WARNING - region_in_exon() - guide at cutsite {cutsite}: found no exons")
            return False 
        
        if len(exons) != 1: # guide must be in exactly one exon
            # print(f"WARNING - region_in_exon() - guide at cutsite {cutsite}: found more than 1 exon")
            # print(f"\t{exons}")
            return False 

        curr_exon = exons[0]

        cutsite_buffer = 3
        if (cutsite - cutsite_buffer >= curr_exon.start) and (cutsite + cutsite_buffer <= curr_exon.stop):
            
            # Cut is good! Now find which exon it is in
            exon_ct = 0
            gene_name = f"gene:{gene}" # gff is weird?
            gene = gff[gene_name]
        
            for exon in gff.children(gene, featuretype='exon', order_by="seqid"):
                exon_ct += 1
                
                # cutsite is in this exon
                if exon.start <= cutsite <= exon.stop:
                    return exon_ct
            

        # print(f"WARNING - region_in_exon() - guide at cutsite {cutsite}: cutsite not in exon")
        return False


    def get_cutsite(self, start, end, strand):

        if strand == "+":
            return end - 6
        elif strand == "-":
            return start + 6
        return np.nan
    

    def annotate_variants(self, df):
        

        guides = self.make_bed(df)

        snps   = BedTool(self.variants)
        
        # only report things from guides if they intercept with things from snps
        intersect = guides.intersect(snps, u=True)

        intersect_df = intersect.to_dataframe()
        
        # sloppy af... but whatevs
        if len(intersect_df) == 0: # no guides with snps
            bad_guides = []

            logging.debug("found no snps")
        else:
            bad_guides = list(intersect_df["score"])

            logging.debug(f"found {len(bad_guides)} guides with snps")

        df["has_variant"] = np.nan # default unknown value


        guides_checked = list(guides.to_dataframe()["score"]) # target_seq

        df.loc[df['targetSeq'].isin(guides_checked), 'has_variant'] = False     # if guide was checked for snp
        
        df.loc[df['targetSeq'].isin(bad_guides), 'has_variant'] = True  # if guide was checked for snp and snp found

        return df
   

    def blast_guides(self, gene, gff, df, outdir):
        """ df must have guideId and targetseq """
        df_fasta = df[["guideId", "targetSeq"]].copy()
        df_fasta["guideId"] = df_fasta['guideId'].map(lambda s: f">{s}")

        os.makedirs(outdir, exist_ok=True)

        tmp_fasta = os.path.join(outdir, gene + ".input")
        df_fasta.to_csv(tmp_fasta, index=False, sep="\n")

        subprocess.run(f"ml ncbi-blast+", shell=True, capture_output=True)

        tmp_blast = os.path.join(outdir, gene + ".blast")

        cmd = f"blastn -query {tmp_fasta} -db resources/Phaeodactylum_tricornutum.ASM15095v2.dna.toplevel.fa -task blastn-short -evalue 0.0000035 -dust no -soft_masking false -outfmt 6 > {tmp_blast}"
        subprocess.run(cmd, shell=True, capture_output=True)
        
        df_blast = pd.read_csv(tmp_blast, sep='\t', header=None, names=["guideId", "chrom", "ident", "match_len", "mismatch", "gopen",  "qstart", "qstop", "start",  "stop", "eval", "bit"])

        # check only perfect blast results
        df_blast = df_blast[(df_blast.match_len == 23) & (df_blast.mismatch == 0)].copy()

        # make bed
        df_bed  = df_blast[["chrom", "start", "stop", "guideId", "eval"]].copy()

        df_bed["strand"] = np.where(df_bed['stop'] - df_bed['start'] > 0, '+', '-')
        df_bed["start"]  = np.minimum(df_blast.start, df_blast.stop).astype(int)
        df_bed["stop"]  = np.maximum(df_blast.start, df_blast.stop).astype(int)


        # df_blast["strand"] = df_bed["strand"]
        # df_blast["start"] = df_bed["start"]
        # df_blast["stop"] = df_bed["stop"]


        # blast_before = os.path.join(outdir, gene + "blast_before.tsv")
        # df_blast.to_csv(blast_before, sep="\t", header=True)

        gene_feature = gff[f"gene:{gene}"]

        # 50 for flank... bad
        df_bed = df_bed[(df_bed.chrom.astype(str) == gene_feature.chrom) & (df_bed.start >= (gene_feature.start - 50)) & (df_bed.stop <= (gene_feature.stop + 50))].copy()


        pre_drop_len = len(df_bed)
        df_bed = df_bed.drop_duplicates(subset=['guideId'], keep=False)

        # Make sure no sequences found twice in gene...
        if pre_drop_len != len(df_bed):
            logging.error(f"Found multiple possible locations for one or more guides in {gene}. Setting location of failed guides to NA")

        if len(df_bed) != len(df_fasta):
            logging.error(f"Failed to identify the location of {len(df_fasta) - len(df_bed)} guide(s) designed for {gene}. Setting location of failed guides to NA")

        # df_blast = df_blast[(df_blast.chrom.astype(str) == gene_feature.chrom) & (df_blast.start >= (gene_feature.start - 50)) & (df_blast.stop <= (gene_feature.stop + 50))].copy()

        # blast_after = os.path.join(outdir, gene + "blast_after.tsv")
        # df_blast.to_csv(blast_after, sep="\t", header=True)


        # print("ASdf")
        # print(df_bed)
        
        bed_file = os.path.join(outdir, gene + ".bed")
        df_bed.to_csv(bed_file, sep='\t', index=False)
       
        df_bed.drop(columns=["eval"], inplace=True) # just clutter

        merged_df = pd.merge(df, df_bed, on="guideId", how="left")

        # types were changing... ?
        merged_df["start"] = merged_df["start"].astype('Int64')
        merged_df["stop"] = merged_df["stop"].astype('Int64')

        return merged_df

    def make_bed(self, df):
        

        df_fasta = df[["chrom", "start", "stop", "seqId", "targetSeq", "strand"]]

        df_fasta = df_fasta.dropna()

        return pybedtools.BedTool.from_dataframe(df_fasta)


def parse_args():

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

    parser.add_argument('-t',
                        help='number cores to use',
                        type=int,
                        default=4,
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

    parser.add_argument('--gff',
                        help='gff3 for genome to use for guide design',
                        default="resources/Phaeodactylum_tricornutum.ASM15095v2.52.gff3",
                        type=str,
                        required=False)
    
    parser.add_argument('--bed',
                        help='bed for genome to use for guide design',
                        default="resources/Phaeodactylum_tricornutum.ASM15095v2.52.bed",
                        type=str,
                        required=False)
    
    parser.add_argument('--variants',
                        help='variants to account for in guide design',
                        default="resources/wt.filt.vcf.recode.vcf",
                        type=str,
                        required=False)
    
    parser.add_argument('--chrom_sizes',
                        help='chrom_sizes for bedtools',
                        default="resources/Phaeodactylum_tricornutum.ASM15095v2.dna.chrom.sizes",
                        type=str,
                        required=False)

    args = parser.parse_args()

    return args
    

if __name__ == "__main__":

    args = parse_args()

    print("Running Guide Designer")
    print("MULTIPROCESSING")

    profiler = cProfile.Profile()
    profiler.enable()

    designer = Guide_Designer(args)
    designer.run()

    profiler.disable()
    profiler.print_stats(sort="tottime")
    print("MULTIPROCESSING")

