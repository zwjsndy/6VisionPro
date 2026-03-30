import ipaddress
import argparse

parser = argparse.ArgumentParser(description="Separate active and aliased addresses.")
parser.add_argument("--aliased_file_path", type=str, required=True,
                    help="Aliased prefix file prefix (e.g. 'aliased_res'). "
                         "Files: {prefix}_{i}.txt")
parser.add_argument("--active_file_path", type=str, required=True,
                    help="Active probe output prefix (e.g. 'active_output'). "
                         "Files: {prefix}_{i}.txt")
parser.add_argument("--active_final_file_path", type=str, required=True,
                    help="Output active file prefix (e.g. 'active_final'). "
                         "Files: {prefix}_{i}.txt")
parser.add_argument("--aliased_final_file_path", type=str, required=True,
                    help="Output aliased file prefix (e.g. 'aliased_final'). "
                         "Files: {prefix}_{i}.txt")
parser.add_argument("--k", type=int, default=6,
                    help="Number of clusters (default: 6).")
args = parser.parse_args()

total_all = 0
active_all = 0
aliased_all = 0

for i in range(args.k):
    print(f"\n{'=' * 50}")
    print(f"  Cluster {i}")
    print(f"{'=' * 50}")

    aliased_prefixes = []
    with open(f"{args.aliased_file_path}_{i}.txt") as f:
        for line in f:
            prefix = line.strip()
            if not prefix:
                continue
            try:
                net = ipaddress.IPv6Network(prefix + "/64", strict=False)
                aliased_prefixes.append(net)
            except ValueError:
                print("Skipping invalid line:", prefix)

    print(f"  Loaded {len(aliased_prefixes)} aliased prefixes")

    total = 0
    active_kept = 0
    aliased_found = 0

    with open(f"{args.active_file_path}_{i}.txt") as infile, \
         open(f"{args.active_final_file_path}_{i}.txt", "w") as active_out, \
         open(f"{args.aliased_final_file_path}_{i}.txt", "w") as aliased_out:

        for line in infile:
            addr = line.strip()
            if not addr:
                continue
            total += 1
            try:
                ip = ipaddress.IPv6Address(addr)
            except ValueError:
                print("Skipping invalid IP:", addr)
                continue

            if any(ip in net for net in aliased_prefixes):
                aliased_out.write(addr + "\n")
                aliased_found += 1
            else:
                active_out.write(addr + "\n")
                active_kept += 1

    filter_pct = round(aliased_found / total * 100, 2) if total > 0 else 0
    print(f"  Total: {total} | Active: {active_kept} | "
          f"Aliased: {aliased_found} | Filter ratio: {filter_pct}%")

    total_all += total
    active_all += active_kept
    aliased_all += aliased_found

print(f"\n{'=' * 50}")
print(f"OVERALL SUMMARY")
print(f"{'=' * 50}")
print(f"Total: {total_all}")
print(f"Active: {active_all}")
print(f"Aliased: {aliased_all}")
if total_all > 0:
    print(f"Filter ratio: {round(aliased_all / total_all * 100, 2)}%")