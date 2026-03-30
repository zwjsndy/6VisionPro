import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipaddress
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 0. Command-line arguments
# ==========================================
parser = argparse.ArgumentParser(
    description="Generate IPv6 addresses from trained GatedPixelCNN models. "
                "Supports any stack size including 1 (no context ablation).")
parser.add_argument("--k", type=int, default=6,
                    help="Number of clusters (default: 6).")
parser.add_argument("--budget", type=int, required=True,
                    help="Total generation budget across all clusters.")
parser.add_argument("--budget_per_cluster", type=int, default=None,
                    help="Budget per cluster (overrides even split).")
parser.add_argument("--stack_size", type=int, default=6,
                    help="Stack size matching trained model (default: 6).")
parser.add_argument("--exclude_file", type=str, nargs="+", default=None,
                    help="Txt files of previously generated addresses to exclude.")
parser.add_argument("--model_name_prefix", type=str, default="cluster",
                    help="Model filename prefix (default: 'cluster'). "
                         "Model path: {prefix}_{cl}_with_1_{s}_{w}.pth. "
                         "For RL-finetuned models, use e.g. 'cluster' with "
                         "--model_name_suffix RL_finetuned.")
parser.add_argument("--model_name_suffix", type=str, default="",
                    help="Optional suffix inserted after cluster id (default: none). "
                         "E.g. --model_name_suffix RL_finetuned loads "
                         "cluster_{cl}_RL_finetuned_with_1_6_128.pth.")
parser.add_argument("--seed_file", type=str, required=True,
                    help="Path to seed file.")
parser.add_argument("--label_file", type=str, default="./label.txt",
                    help="Path to label file (default: ./label.txt).")
args = parser.parse_args()

K_VALUE = args.k
TOTAL_BUDGET = args.budget

# ==========================================
# 1. Configuration
# ==========================================
SEED_FILE = args.seed_file
MODEL_DIR = "./model"
OUTPUT_DIR = "./temp"

STACK_SIZE = args.stack_size
CONTEXT_ROWS = STACK_SIZE - 1  # 5+1 for stack=6, 0+1 for stack=1
H, W = STACK_SIZE, 128
GEN_BATCH_SIZE = 256
OVERSHOOT_RATIO = 1.05

if args.budget_per_cluster is not None:
    BUDGET_PER_CLUSTER = args.budget_per_cluster
else:
    even_split = TOTAL_BUDGET // K_VALUE
    BUDGET_PER_CLUSTER = int(even_split * OVERSHOOT_RATIO)

print(f">>> k={K_VALUE}, total_budget={TOTAL_BUDGET}, "
      f"budget_per_cluster={BUDGET_PER_CLUSTER}")
print(f"    stack_size={STACK_SIZE}, context_rows={CONTEXT_ROWS}, "
      f"scheme: {CONTEXT_ROWS}+1")

# ==========================================
# 2. GPU setup
# ==========================================
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS == 0:
    raise RuntimeError("No GPU detected.")

gpu_names = [torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]
print(f">>> Detected {NUM_GPUS} GPU(s): {gpu_names}")
USE_THREADS = NUM_GPUS >= 2
if USE_THREADS:
    print(f">>> Multi-GPU: dynamic pool across {NUM_GPUS} GPUs")
else:
    print(f">>> Single GPU: sequential execution")

# ==========================================
# 3. Network architecture (fixed MaskedConv2d)
# ==========================================
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer('mask', mask[None, None])
    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

class VerticalStackConv(MaskedConv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        self.mask_type = mask_type
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        mask = torch.zeros(kernel_size)
        mask[:kernel_size[0]//2, :] = 1.0
        if self.mask_type == "B": mask[kernel_size[0]//2, :] = 1.0
        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)

class HorizontalStackConv(MaskedConv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        self.mask_type = mask_type
        if isinstance(kernel_size, int): kernel_size = (1, kernel_size)
        assert kernel_size[0] == 1
        if "padding" in kwargs and isinstance(kwargs["padding"], int):
            kwargs["padding"] = (0, kwargs["padding"])
        mask = torch.zeros(kernel_size)
        mask[:, :kernel_size[1]//2] = 1.0
        if self.mask_type == "B": mask[:, kernel_size[1]//2] = 1.0
        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)

class GatedMaskedConv(nn.Module):
    def __init__(self, in_channels, kernel_size=(3, 15)):
        super().__init__()
        v_padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv_vert = VerticalStackConv(
            "B", in_channels, 2*in_channels, kernel_size, padding=v_padding)
        h_kernel = (1, kernel_size[1])
        h_padding = (0, kernel_size[1] // 2)
        self.conv_horiz = HorizontalStackConv(
            "B", in_channels, 2*in_channels, h_kernel, padding=h_padding)
        self.conv_vert_to_horiz = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1)
        self.conv_horiz_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, v_stack, h_stack):
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack
        return v_stack_out, h_stack_out

class GatedPixelCNN(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        self.conv_vstack = VerticalStackConv(
            "A", in_channels, channels, (3, 15), padding=(1, 7))
        self.conv_hstack = HorizontalStackConv(
            "A", in_channels, channels, (1, 15), padding=(0, 7))
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(channels, kernel_size=(3, 15)) for _ in range(7)
        ])
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=1)
    def forward(self, x):
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        return self.conv_out(F.elu(h_stack))

# ==========================================
# 4. Utility functions
# ==========================================
def convert(seeds):
    result = []
    for line in seeds:
        line = line.strip().split(":")
        if not line or line == ['']: continue
        for i in range(len(line)):
            if len(line[i]) == 4: continue
            if len(line[i]) < 4 and len(line[i]) > 0:
                line[i] = line[i].zfill(4)
            if len(line[i]) == 0:
                line[i] = "0000" * (9 - len(line))
        result.append("".join(line)[:32])
    return result

def hex2two(a):
    state_10 = int(a, 16)
    str1 = '{:04b}'.format(state_10)
    return '0' * (len(4 * a) - len(str1)) + str1

def str2ipv6(a):
    pattern = re.compile('.{4}')
    return str(ipaddress.ip_address(':'.join(pattern.findall(a))))

def addr_to_128bits(hex_str):
    return [float(b) for b in hex2two(hex_str)]

# ==========================================
# 5. Load shared data (seeds, labels)
# ==========================================
print(f"Loading seed file: {SEED_FILE} ...")
raw_seeds = []
exclusion_set = set()
with open(SEED_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        l = line.strip()
        if l:
            raw_seeds.append(l)
            try:
                exclusion_set.add(str(ipaddress.ip_address(l)))
            except:
                pass
print(f"Original seed pool: {len(exclusion_set)} addresses")

# Exclude files
if args.exclude_file:
    total_prev = 0
    for exc_path in args.exclude_file:
        if os.path.exists(exc_path):
            file_count = 0
            with open(exc_path, 'r', encoding='utf-8') as f:
                for line in f:
                    addr = line.strip()
                    if addr:
                        try:
                            exclusion_set.add(str(ipaddress.ip_address(addr)))
                            file_count += 1
                        except:
                            pass
            print(f"  Loaded {file_count} from {exc_path}")
            total_prev += file_count
        else:
            print(f"  Warning: not found: {exc_path}")
    if total_prev > 0:
        print(f"  Combined exclusion set: {len(exclusion_set)}")

res_list = convert(raw_seeds)

label_file = args.label_file
if not os.path.exists(label_file):
    raise FileNotFoundError(f"Label file not found: {label_file}.")

print(f"Loading label file: {label_file} ...")
labels = []
with open(label_file, 'r', encoding='utf-8') as f:
    for line in f:
        l = line.strip()
        if l: labels.append(int(l))

if len(res_list) != len(labels):
    min_len = min(len(res_list), len(labels))
    res_list = res_list[:min_len]
    labels = labels[:min_len]

cluster_pools = {}
for i in range(K_VALUE):
    pool = [res_list[j] for j, lb in enumerate(labels) if lb == i]
    cluster_pools[i] = pool
    print(f"  Cluster {i}: {len(pool)} seeds")

# ==========================================
# 6. Single-cluster generation function
# ==========================================
def generate_for_cluster(cluster_idx, gpu_id, budget, cluster_pool,
                         exclusion_set_local):
    device = torch.device(f"cuda:{gpu_id}")

    suffix = f"_{args.model_name_suffix}" if args.model_name_suffix else ""
    model_path = os.path.join(
        MODEL_DIR,
        f"{args.model_name_prefix}_{cluster_idx}{suffix}_with_1_{STACK_SIZE}_{W}.pth")
    if not os.path.exists(model_path):
        print(f"  [Cluster {cluster_idx}] Model not found: {model_path}, skipping.")
        return []

    model = torch.load(model_path, map_location=device,
                       weights_only=False, mmap=False)
    model = model.to(device)
    model.eval()

    results = []
    local_seen = set()
    duplicate_hits = 0
    batch_count = 0
    batch_start = time.time()

    while len(results) < budget:
        pixels = torch.zeros(GEN_BATCH_SIZE, 1, H, W, device=device)
        context_batch = []

        # Fill context rows (0 for stack_size=1, 5 for stack_size=6)
        if CONTEXT_ROWS > 0:
            for b in range(GEN_BATCH_SIZE):
                neighbors = np.random.choice(cluster_pool, CONTEXT_ROWS, replace=True)
                context_batch.append(list(neighbors))
                for r_idx, addr_hex in enumerate(neighbors):
                    pixels[b, 0, r_idx, :] = torch.tensor(
                        addr_to_128bits(addr_hex), device=device)
        else:
            for b in range(GEN_BATCH_SIZE):
                context_batch.append([])

        # Generate target row (last row) bit-by-bit
        target_row = H - 1
        with torch.no_grad():
            for w in range(W):
                logits = model(pixels)[:, 0, target_row, w]
                probs = torch.sigmoid(logits)
                pixels[:, 0, target_row, w] = torch.bernoulli(probs)

        # Decode and filter
        generated_rows = pixels[:, 0, target_row, :].cpu().numpy()

        for b in range(GEN_BATCH_SIZE):
            if len(results) >= budget:
                break
            bit_str = "".join([str(int(bit)) for bit in generated_rows[b]])
            try:
                hex_str = hex(int(bit_str, 2))[2:].zfill(32)
                addr = str2ipv6(hex_str)

                if addr in exclusion_set_local:
                    duplicate_hits += 1
                    continue

                if addr not in local_seen:
                    local_seen.add(addr)
                    results.append({
                        "context_hex": context_batch[b],
                        "generated_hex": hex_str,
                        "generated_ipv6": addr,
                        "cluster": cluster_idx,
                    })
            except Exception:
                continue

        # Real-time progress
        batch_count += 1
        elapsed = time.time() - batch_start
        speed = len(results) / elapsed if elapsed > 0 else 0
        eta = (budget - len(results)) / speed if speed > 0 else 0
        pct = 100.0 * len(results) / budget
        print(f"\r    [Cluster {cluster_idx}] "
              f"{len(results):>6d}/{budget} ({pct:5.1f}%) | "
              f"{speed:.0f} addr/s | "
              f"ETA {eta:.0f}s | "
              f"batch {batch_count} | "
              f"dupes {duplicate_hits}",
              end="", flush=True)

    print()
    del model
    torch.cuda.empty_cache()

    print(f"  [Cluster {cluster_idx} @ GPU {gpu_id}] "
          f"Generated {len(results)} addresses, "
          f"{duplicate_hits} dupes skipped")
    return results

# ==========================================
# 7. Generation
# ==========================================
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"\n{'=' * 60}")
print(f">>> Starting generation: {K_VALUE} clusters, "
      f"{NUM_GPUS} GPU(s), {BUDGET_PER_CLUSTER} per cluster")
print(f"    Scheme: {CONTEXT_ROWS}+1 (stack_size={STACK_SIZE})")
print(f"{'=' * 60}")

all_cluster_results = {}
start_time = time.time()

if USE_THREADS:
    import queue
    gpu_queue = queue.Queue()
    for g in range(NUM_GPUS):
        gpu_queue.put(g)

    def generate_with_gpu_pool(cluster_idx, budget, cluster_pool,
                               exclusion_set_local):
        gpu_id = gpu_queue.get()
        try:
            return generate_for_cluster(
                cluster_idx, gpu_id, budget,
                cluster_pool, exclusion_set_local)
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = {}
        for cl_idx in range(K_VALUE):
            if not cluster_pools[cl_idx]:
                continue
            future = executor.submit(
                generate_with_gpu_pool,
                cl_idx, BUDGET_PER_CLUSTER,
                cluster_pools[cl_idx], exclusion_set)
            futures[future] = cl_idx

        for future in as_completed(futures):
            cl_idx = futures[future]
            try:
                all_cluster_results[cl_idx] = future.result()
            except Exception as e:
                print(f"  [Cluster {cl_idx}] Failed: {e}")
                all_cluster_results[cl_idx] = []
else:
    for cl_idx in range(K_VALUE):
        if not cluster_pools[cl_idx]:
            continue
        cl_start = time.time()
        results = generate_for_cluster(
            cl_idx, 0, BUDGET_PER_CLUSTER,
            cluster_pools[cl_idx], exclusion_set)
        all_cluster_results[cl_idx] = results
        print(f"  [Cluster {cl_idx}] Done in {time.time()-cl_start:.1f}s")

elapsed = time.time() - start_time
print(f"\nGeneration phase complete in {elapsed:.1f}s")

# ==========================================
# 8. Post-hoc cross-cluster deduplication
# ==========================================
print(f"\n>>> Post-hoc deduplication...")
global_seen = set(exclusion_set)
deduped_results = {}
total_before = 0
total_after = 0

for cl_idx in range(K_VALUE):
    if cl_idx not in all_cluster_results:
        continue
    raw = all_cluster_results[cl_idx]
    total_before += len(raw)
    deduped = []
    for item in raw:
        addr = item["generated_ipv6"]
        if addr not in global_seen:
            global_seen.add(addr)
            deduped.append(item)
    deduped_results[cl_idx] = deduped
    total_after += len(deduped)
    removed = len(raw) - len(deduped)
    print(f"  Cluster {cl_idx}: {len(raw)} -> {len(deduped)} "
          f"({removed} cross-cluster dupes)")

print(f"\n  Total: {total_before} -> {total_after} "
      f"({total_before - total_after} removed)")

# ==========================================
# 9. Trim to budget and save
# ==========================================
budget_per_cluster_final = TOTAL_BUDGET // K_VALUE
remainder = TOTAL_BUDGET % K_VALUE

print(f"\n>>> Trimming to {TOTAL_BUDGET} "
      f"(~{budget_per_cluster_final} per cluster)")

total_saved = 0
saved_files = []

for cl_idx in range(K_VALUE):
    if cl_idx not in deduped_results:
        continue
    trim_target = budget_per_cluster_final + (1 if cl_idx < remainder else 0)
    cluster_data = deduped_results[cl_idx][:trim_target]

    txt_path = os.path.join(
        OUTPUT_DIR,
        f"res_{cl_idx}_with_1_{STACK_SIZE}_{W}.txt")
    jsonl_path = os.path.join(
        OUTPUT_DIR,
        f"generated_log_res_{cl_idx}_with_1_{STACK_SIZE}_{W}.jsonl")

    with open(txt_path, 'w') as f_txt, open(jsonl_path, 'w') as f_jsonl:
        for item in cluster_data:
            f_txt.write(item["generated_ipv6"] + '\n')
            f_jsonl.write(json.dumps(item) + '\n')

    total_saved += len(cluster_data)
    saved_files.append((cl_idx, len(cluster_data), txt_path, jsonl_path))
    print(f"  Cluster {cl_idx}: saved {len(cluster_data)} addresses")

print(f"\n{'=' * 60}")
print(f">>> All done! Total: {total_saved}/{TOTAL_BUDGET}")
print(f"{'=' * 60}")
for cl_idx, count, txt_path, _ in saved_files:
    print(f"  Cluster {cl_idx} ({count}): {txt_path}")
print(f"\nUsage: python generate.py --budget {TOTAL_BUDGET} "
      f"--stack_size {STACK_SIZE}")