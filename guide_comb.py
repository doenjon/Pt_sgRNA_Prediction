
from itertools import product

guides_file = "all_guides.fasta"
guides = {}

with open(guides_file, "r") as f:
    for line in f:
        if line.startswith(">"):
            id = line[1:].strip()
            guide = f.readline().strip()
            
            guides[guide] = id

seq_file = "seq.fasta"
seq = []
with open(seq_file, "r") as f:
    for line in f:
        chars = line.strip().split()
        chars = [c.upper() for c in chars]
        seq.append(chars)

print(seq)

print(list(guides.keys())[1:10])

def get_guides(seq, guides, print_out=False):
    found_guides = []
    for s in product(*seq):
        s = "".join(s)
        # print(s)
        for guide in guides:
            if s in guide:
                if print_out:
                    print(s)
                found_guides.append(guide)
    return found_guides

round0_guides = get_guides(seq[10:20], list(guides.keys()))
print(len(round0_guides))

round1_guides = get_guides(seq[0:14], round0_guides)
print(len(round1_guides))

round2_guides = get_guides(seq, round1_guides, True)

for g in round2_guides:
    print(g)
    print(guides[g])
    print()



# candidate_guides = []
# small_seq = seq[0:10]
# for s in product(*small):
#     s = "".join(s)
#     # print(s)
#     for guide in guides.keys():
#         if s in guide:
#             print(s)
#             print(guide)
#             print()
   
    # if s in guides.keys():
    #     print(s)
    #     print(guides[s])
    #     print()
