import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ipaddress
import re
import time

# ==========================================
# 0. Command-line arguments
# ==========================================
parser = argparse.ArgumentParser(description="Train GatedPixelCNN per cluster")
parser.add_argument("--k", type=int, default=6,
                    help="Number of clusters (default: 6).")
parser.add_argument("--stack_size", type=int, default=6,
                    help="Number of rows in the input stack (default: 6).")
parser.add_argument("--seed_file", type=str, required=True,
                    help="Path to seed file")
parser.add_argument("--label_file", type=str, default="./label.txt",
                    help="Path to label file (default: ./label.txt).")
args = parser.parse_args()

K_VALUE = args.k

# ==========================================
# 1. Configuration
# ==========================================
SEED_FILE = args.seed_file
LABEL_FILE = args.label_file
MODEL_DIR = "./model"

STACK_SIZE = args.stack_size
H, W = STACK_SIZE, 128
IN_CHANNELS = 1
OUT_CHANNELS = 1
CHANNELS = 128
BATCH_SIZE = 64
EPOCHS = 30
LR = 5e-4
EPOCH_MULTIPLIER = 3
MIN_CLUSTER_SIZE = STACK_SIZE
EARLY_STOP_LOSS = 0.05
NUM_GATED_LAYERS = 7
KERNEL_SIZE = (3, 15)
NUM_WORKERS = 4

print(f">>> Stack size: {STACK_SIZE} (input shape: 1x{H}x{W})")

# ==========================================
# Multi-GPU setup
# ==========================================
NUM_GPUS = torch.cuda.device_count()
USE_MULTI_GPU = NUM_GPUS >= 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if NUM_GPUS > 0:
    gpu_names = [torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]
    print(f">>> Detected {NUM_GPUS} GPU(s): {gpu_names}")
    if USE_MULTI_GPU:
        print(f"    Using DataParallel across {NUM_GPUS} GPUs")
        BATCH_SIZE = BATCH_SIZE * NUM_GPUS
        print(f"    Batch size scaled to {BATCH_SIZE}")
else:
    print(">>> No GPU detected, using CPU")

if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(
        f"Label file not found: {LABEL_FILE}\n"
        f"Run clustering first to generate label files.")

print(f">>> Using k={K_VALUE}, label file: {LABEL_FILE}")

# ==========================================
# 2. Network architecture (fixed MaskedConv2d)
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
        self.conv_vert = VerticalStackConv("B", in_channels, 2*in_channels, kernel_size, padding=v_padding)
        
        h_kernel = (1, kernel_size[1]) 
        h_padding = (0, kernel_size[1] // 2) 
        self.conv_horiz = HorizontalStackConv("B", in_channels, 2*in_channels, h_kernel, padding=h_padding)
        
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
        self.conv_vstack = VerticalStackConv("A", in_channels, channels, KERNEL_SIZE, padding=(1, 7))
        self.conv_hstack = HorizontalStackConv("A", in_channels, channels, (1, KERNEL_SIZE[1]), padding=(0, 7))
        
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(channels, kernel_size=KERNEL_SIZE) for _ in range(NUM_GATED_LAYERS)
        ])
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        out = self.conv_out(F.elu(h_stack))
        return out

# ==========================================
# 3. Utility functions
# ==========================================

def convert(seeds):
    """Normalize IPv6 addresses to 32-char hex strings."""
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
        hex_str = "".join(line)
        result.append(hex_str[:32])
    return result

def hex2two(a):
    state_10 = int(a, 16)
    str1 = '{:04b}'.format(state_10)
    res = '0' * (len(4 * a) - len(str1)) + str1
    return res

def addr_to_128bits(hex_str):
    """Convert a 32-char hex string to a list of 128 binary floats."""
    b_str = hex2two(hex_str)
    return [float(b) for b in b_str]

# ==========================================
# 4. Dataset definition
# ==========================================

class IPv6StackDataset(Dataset):
    def __init__(self, seeds_list_128bit, stack_size=6, epoch_multiplier=3):
        self.data_tensor = torch.tensor(seeds_list_128bit, dtype=torch.float32)
        self.num_seeds = len(seeds_list_128bit)
        self.stack_size = stack_size
        self.virtual_len = self.num_seeds * epoch_multiplier

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx):
        indices = torch.randint(0, self.num_seeds, (self.stack_size,))
        stack = self.data_tensor[indices]
        stack = stack.unsqueeze(0)
        return stack, stack

# ==========================================
# 5. Main program
# ==========================================

if __name__ == "__main__":
    print(f"Loading seed file: {SEED_FILE} ...")
    raw_seeds = []
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                l = line.strip()
                if l: raw_seeds.append(l)
    else:
        raise FileNotFoundError(f"Seed file not found: {SEED_FILE}")

    print(f"Total raw seeds: {len(raw_seeds)}")
    res_hex = convert(raw_seeds)

    print("Preprocessing address features...")
    res_128bits = [addr_to_128bits(addr) for addr in res_hex]

    print(f"Loading label file: {LABEL_FILE} ...")
    labels = []
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                l = line.strip()
                if l: labels.append(int(l))
    else:
        raise FileNotFoundError(f"Label file not found: {LABEL_FILE}")

    if len(res_128bits) != len(labels):
        print(f"Warning: seed count ({len(res_128bits)}) != label count ({len(labels)}), truncating.")
        min_len = min(len(res_128bits), len(labels))
        res_128bits = res_128bits[:min_len]
        labels = labels[:min_len]

    all_data = [] 
    max_label = max(labels)
    for i in range(max_label + 1):
        cluster_seeds = [res_128bits[j] for j, lb in enumerate(labels) if lb == i]
        all_data.append(cluster_seeds)

    print(f"Grouping complete: {len(all_data)} clusters (k={K_VALUE}).")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    for cl_idx, cluster_pool in enumerate(all_data):
        if len(cluster_pool) < MIN_CLUSTER_SIZE:
            print(f"Cluster {cl_idx} has too few samples (<{MIN_CLUSTER_SIZE}), skipping.")
            continue

        print(f"\n>>> Training cluster {cl_idx} (seed count: {len(cluster_pool)})")
        
        train_dataset = IPv6StackDataset(
            cluster_pool, stack_size=STACK_SIZE, epoch_multiplier=EPOCH_MULTIPLIER)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS, 
            pin_memory=True
        )

        model = GatedPixelCNN(IN_CHANNELS, CHANNELS, OUT_CHANNELS)
        if USE_MULTI_GPU:
            model = nn.DataParallel(model)
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            start_time = time.time()
            
            for i, (images, _) in enumerate(train_loader):
                images = images.to(DEVICE, non_blocking=True)
                
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, images)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if (i + 1) % 10 == 0:
                    print(f"\rEpoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end="")
            
            avg_loss = total_loss / len(train_loader)
            epoch_duration = time.time() - start_time
            print(f"\rCluster {cl_idx} | Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f} | Time: {epoch_duration:.1f}s")
            
            if avg_loss < EARLY_STOP_LOSS:
                print("Loss sufficiently low, stopping early.")
                break

        save_path = os.path.join(
            MODEL_DIR,
            f'cluster_{cl_idx}_with_1_{STACK_SIZE}_{W}.pth'
        )
        model_to_save = model.module if USE_MULTI_GPU else model
        torch.save(model_to_save, save_path)
        print(f"Model saved: {save_path}")

    print(f"\nAll training tasks complete! (num_clusters={K_VALUE}, stack_size={STACK_SIZE})")