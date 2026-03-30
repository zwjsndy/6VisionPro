import csv
import argparse

parser = argparse.ArgumentParser(description="Extract active addresses from ZMap output.")
parser.add_argument("--input_file_path", type=str, required=True,
                    help="Input file path and prefix. "
                         "Files: {prefix}_{i}.txt for i in 0..k-1.")
parser.add_argument("--output_file_path", type=str, required=True,
                    help="Output file path and prefix. "
                         "Files: {prefix}_{i}.txt for i in 0..k-1.")
parser.add_argument("--k", type=int, default=6,
                    help="Number of clusters (default: 6).")
args = parser.parse_args()

for i in range(args.k):
    with open(f"{args.input_file_path}_{i}.txt", newline="") as infile, \
         open(f"{args.output_file_path}_{i}.txt", "w") as outfile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row["classification"] == "echo_reply":
                outfile.write(row["source_addr"] + "\n")