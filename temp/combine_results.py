import os
import argparse

parser = argparse.ArgumentParser(
    description="Combine per-cluster result txt files into a single file.")
parser.add_argument("--k", type=int, default=6,
                    help="Number of clusters (default: 6).")
parser.add_argument("--stack_size", type=int, default=6,
                    help="Stack size to identify correct files (default: 6).")
parser.add_argument("--input_dir", type=str, default="./",
                    help="Directory containing result files (default: ./).")
parser.add_argument("--result_name", type=str, default="res",
                    help="Filename prefix: 'res' for first-round, 'RL_res' for post-RL (default: res).")
parser.add_argument("--output", type=str, default=None,
                    help="Output file path (default: auto-generated).")
args = parser.parse_args()

K_VALUE = args.k
STACK_SIZE = args.stack_size
W = 128
INPUT_DIR = args.input_dir
PREFIX = args.result_name

# Auto-generate output filename if not specified
if args.output:
    output_path = args.output
else:
    output_path = os.path.join(
        INPUT_DIR,
        f"combined_{PREFIX}_k{K_VALUE}_s{STACK_SIZE}.txt"
    )

total = 0
found = 0

with open(output_path, 'w', encoding='utf-8') as out_f:
    for cl_idx in range(K_VALUE):
        # Try the standard naming pattern
        txt_path = os.path.join(
            INPUT_DIR,
            f"{PREFIX}_{cl_idx}_with"
            f"_1_{STACK_SIZE}_{W}.txt"
        )
        # Also try the RL short naming pattern
        if not os.path.exists(txt_path):
            txt_path = os.path.join(
                INPUT_DIR,
                f"{PREFIX}_{cl_idx}_k{K_VALUE}_s{STACK_SIZE}.txt"
            )

        if not os.path.exists(txt_path):
            print(f"  Cluster {cl_idx}: not found, skipping.")
            continue

        count = 0
        with open(txt_path, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                addr = line.strip()
                if addr:
                    out_f.write(addr + '\n')
                    count += 1

        total += count
        found += 1
        print(f"  Cluster {cl_idx}: {count} addresses")

print(f"\nCombined {found}/{K_VALUE} clusters, {total} total addresses")
print(f"Saved to: {output_path}")