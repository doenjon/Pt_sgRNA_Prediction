import sys
import json
import random

def calculate_gc_content(sequence):
    """Calculate GC content of a sequence."""
    gc_count = sequence.count('G') + sequence.count('C')
    return round((gc_count / len(sequence)) * 100, 1)

def find_pam_sites(sequence):
    """Find all PAM (NGG) sites in the sequence."""
    pam_positions = []
    for i in range(len(sequence) - 2):
        if sequence[i+1:i+3] == 'GG':
            pam_positions.append(i)
    return pam_positions

def generate_guide(sequence, pam_position):
    """Generate a guide sequence and its properties for a given PAM site."""
    # Guide sequence is 20 bp upstream of PAM
    guide_start = max(0, pam_position - 20)
    guide_sequence = sequence[guide_start:pam_position]
    
    # Ensure guide is 20bp long
    if len(guide_sequence) < 20:
        return None
        
    # Calculate properties
    gc_content = calculate_gc_content(guide_sequence)
    
    # Calculate simple score based on GC content and position
    # Optimal GC content is between 40-60%
    gc_score = 1.0 - abs(50 - gc_content) / 50
    position_score = 1.0 - (pam_position / len(sequence))  # Prefer guides closer to start
    
    # Simulate off-target count (in real implementation, this would involve alignment)
    off_targets = random.randint(0, 5)
    off_target_score = 1.0 - (off_targets / 5)
    
    # Combined score
    score = (gc_score + position_score + off_target_score) / 3
    
    return {
        "sequence": guide_sequence,
        "position": guide_start,
        "score": round(score, 3),
        "gc_content": gc_content,
        "off_targets": off_targets
    }

def design_guides(input_sequence):
    """Main function to design guide RNAs."""
    # Debug logging
    print("Debug: Received input sequence: {}".format(input_sequence), file=sys.stderr)
    
    # Convert sequence to uppercase and remove any whitespace
    sequence = input_sequence.strip().upper()
    
    # Validate sequence
    if not sequence:
        print("Debug: Empty sequence received", file=sys.stderr)
        return {"guides": [], "error": "Empty sequence"}
    
    if not all(base in 'ATCG' for base in sequence):
        print("Debug: Invalid bases in sequence: {}".format(sequence), file=sys.stderr)
        return {"guides": [], "error": "Invalid sequence - must contain only A, T, C, G"}
    
    # Find all PAM sites
    pam_positions = find_pam_sites(sequence)
    print("Debug: Found {} PAM sites".format(len(pam_positions)), file=sys.stderr)
    
    # Generate guides for each PAM site
    guides = []
    for pos in pam_positions:
        guide = generate_guide(sequence, pos)
        if guide:
            guides.append(guide)
    
    print("Debug: Generated {} valid guides".format(len(guides)), file=sys.stderr)
    
    # Sort guides by score
    guides.sort(key=lambda x: x["score"], reverse=True)
    
    # # Take top 5 guides
    # guides = guides[:5]
    
    return {
        "guides": guides
    }

if __name__ == "__main__":
    try:
        # Read input sequence from stdin
        print("Debug: Reading from stdin...", file=sys.stderr)
        input_sequence = sys.stdin.read().strip()
        print("Debug: Read {} characters".format(len(input_sequence)), file=sys.stderr)
        
        # Generate guides
        result = design_guides(input_sequence)
        
        # Output JSON to stdout
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        print("Debug: Error occurred: {}".format(str(e)), file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
