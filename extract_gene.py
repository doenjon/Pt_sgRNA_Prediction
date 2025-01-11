import gffutils
from gffutils.helpers import asinterval
from gffutils import FeatureDB

db_file = "resources/pt_genome_gff.sql"

# gffutils.create_db(gff_file, db_file, merge_strategy="create_unique", force=True)

db = gffutils.FeatureDB(db_file)

with open("gene.txt", "w") as file:
    for gene in db.features_of_type("gene"):
        gene = gene.id.split(":")[1]
        file.write(f"{gene}\n")