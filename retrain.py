import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import ipaddress
import re
import argparse

# ==========================================
# 0. Command-line arguments
# ==========================================
parser = argparse.ArgumentParser(
    description="Offline RL (REINFORCE) fine-tuning for GatedPixelCNN. "
                "Automatically retrains all clusters (0 to k-1).")
parser.add_argument("--k", type=int, default=6,
                    help="Number of clusters (default: 6).")
parser.add_argument("--stack_size", type=int, default=6,
                    help="Stack size matching the trained model (default: 6).")
parser.add_argument("--model_name_prefix", type=str, default="cluster",
                    help="Model filename prefix (default: 'cluster'). "
                         "Model path: {prefix}_{cl}_with_1_{s}_{w}.pth")
parser.add_argument("--log_name_prefix", type=str, default="generated_log_res",
                    help="Log filename prefix (default: 'generated_log_res'). "
                         "Log path: {prefix}_{cl}_with_1_{s}_{w}.jsonl")
parser.add_argument("--model_save_name", type=str, default="RL_finetuned",
                    help="Save filename prefix (default: 'RL_finetuned'). "
                         "Save path: cluster_{cl}_{prefix}_with_1_{s}_{w}.pth")
parser.add_argument("--active_final_file_path", type=str, default="active_final",
                    help="Active file prefix: {prefix}_{cl}.txt")
parser.add_argument("--aliased_final_file_path", type=str, default="aliased_final",
                    help="Aliased file prefix: {prefix}_{cl}.txt")
parser.add_argument("--seed_prefixes", type=str, default="./bgp_prefixes_from_seed.txt",
                    help="Prefixes from seeds")
args = parser.parse_args()

K_VALUE = args.k
STACK_SIZE = args.stack_size
H, W = STACK_SIZE, 128

# ==========================================
# 1. Configuration
# ==========================================
MODEL_DIR = "./model"
TEMP_DIR = "./temp"

BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-5
KL_COEF = 0.1
ENTROPY_COEF = 0.2
ADVANTAGE_CLIP_MIN = -1.5
ADVANTAGE_CLIP_MAX = 10.0
GRAD_CLIP_NORM = 2.0
BASELINE_MOMENTUM = 0.9
ENTROPY_EARLY_STOP = 10.0

# Reward values
REWARD_ACTIVE = 10.0
REWARD_ALIASED = -5.0
REWARD_OUT_OF_PREFIX = -4.0
REWARD_DEFAULT = -1.0

# Weighted oversampling weights
WEIGHT_ACTIVE = 50.0
WEIGHT_ALIASED = 5.0
WEIGHT_DEFAULT = 1.0

# BGP prefix constraint file (optional)
BGP_PREFIX_FILE = args.seed_prefixes

# ==========================================
# 2. Multi-GPU setup
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
def hex2two(a):
    state_10 = int(a, 16)
    str1 = '{:04b}'.format(state_10)
    return '0' * (len(4 * a) - len(str1)) + str1

def addr_to_128bits(hex_str):
    return [float(b) for b in hex2two(hex_str)]

def load_bgp_prefixes(prefix_file):
    prefixes = []
    if not os.path.exists(prefix_file):
        print(f"  BGP prefix file not found: {prefix_file}, skipping.")
        return prefixes
    with open(prefix_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                prefixes.append(ipaddress.IPv6Network(line, strict=False))
            except ValueError:
                pass
    print(f"  Loaded {len(prefixes)} BGP prefix constraints.")
    return prefixes

def is_in_bgp_prefixes(ipv6_str, prefixes):
    if not prefixes:
        return True
    try:
        addr = ipaddress.IPv6Address(ipv6_str)
        return any(addr in prefix for prefix in prefixes)
    except ValueError:
        return False

# ==========================================
# 5. Offline RL Dataset (adaptive stack size)
# ==========================================
class OfflineRLDataset(Dataset):
    def __init__(self, contexts_list, generated_list, rewards_list,
                 stack_size=6):
        self.num_samples = len(generated_list)
        self.stack_size = stack_size
        self.data_tensor = torch.zeros(
            (self.num_samples, 1, stack_size, 128), dtype=torch.float32)
        self.rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)

        context_rows = stack_size - 1
        print(f"  Building {stack_size}x128 offline RL matrix "
              f"({self.num_samples} samples, {context_rows} context rows)...")
        for i in range(self.num_samples):
            for row in range(min(context_rows, len(contexts_list[i]))):
                self.data_tensor[i, 0, row, :] = torch.tensor(
                    addr_to_128bits(contexts_list[i][row]))
            self.data_tensor[i, 0, stack_size - 1, :] = torch.tensor(
                addr_to_128bits(generated_list[i]))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.rewards_tensor[idx]

# ==========================================
# 6. Per-cluster retrain function
# ==========================================
def retrain_cluster(cl_idx, bgp_prefixes):
    print(f"\n{'=' * 60}")
    print(f">>> Retraining cluster {cl_idx} (k={K_VALUE}, stack_size={STACK_SIZE})")
    print(f"{'=' * 60}")

    # --- Resolve file paths ---
    model_path = os.path.join(
        MODEL_DIR,
        f"{args.model_name_prefix}_{cl_idx}_with_1_{STACK_SIZE}_{W}.pth")
    log_file = os.path.join(
        TEMP_DIR,
        f"{args.log_name_prefix}_{cl_idx}_with_1_{STACK_SIZE}_{W}.jsonl")
    active_file = os.path.join(
        TEMP_DIR, f"{args.active_final_file_path}_{cl_idx}.txt")
    aliased_file = os.path.join(
        TEMP_DIR, f"{args.aliased_final_file_path}_{cl_idx}.txt")

    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}, skipping.")
        return
    if not os.path.exists(log_file):
        print(f"  Log file not found: {log_file}, skipping.")
        return

    print(f"  Model: {model_path}")
    print(f"  Log: {log_file}")
    print(f"  Active: {active_file} {'(found)' if os.path.exists(active_file) else '(not found)'}")
    print(f"  Aliased: {aliased_file} {'(found)' if os.path.exists(aliased_file) else '(not found)'}")

    # --- Load probe results ---
    active_set = set()
    aliased_set = set()
    if os.path.exists(active_file):
        with open(active_file, 'r') as f:
            active_set = set([line.strip() for line in f if line.strip()])
    if os.path.exists(aliased_file):
        with open(aliased_file, 'r') as f:
            aliased_set = set([line.strip() for line in f if line.strip()])

    # --- Read generation log and assign rewards ---
    contexts_all = []
    generated_all = []
    rewards_all = []

    print(f"  Reading generation log and assigning rewards...")
    with open(log_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            gen_ipv6 = data['generated_ipv6']
            contexts_all.append(data.get('context_hex', []))
            generated_all.append(data['generated_hex'])

            if not is_in_bgp_prefixes(gen_ipv6, bgp_prefixes):
                rewards_all.append(REWARD_OUT_OF_PREFIX)
            elif gen_ipv6 in aliased_set:
                rewards_all.append(REWARD_ALIASED)
            elif gen_ipv6 in active_set:
                rewards_all.append(REWARD_ACTIVE)
            else:
                rewards_all.append(REWARD_DEFAULT)

    # --- Diagnostics ---
    hits = rewards_all.count(REWARD_ACTIVE)
    traps = rewards_all.count(REWARD_ALIASED)
    out_of_prefix = rewards_all.count(REWARD_OUT_OF_PREFIX)
    print(f"  Loaded {len(generated_all)} records: "
          f"active={hits}, aliased={traps}, out_of_prefix={out_of_prefix}")
    if hits == 0:
        print("  Warning: 0 active hits! Model will receive no positive feedback.")

    # --- Build dataset with weighted sampling ---
    dataset = OfflineRLDataset(
        contexts_all, generated_all, rewards_all, stack_size=STACK_SIZE)

    sample_weights = []
    for r in rewards_all:
        if r == REWARD_ACTIVE:
            sample_weights.append(WEIGHT_ACTIVE)
        elif r == REWARD_ALIASED:
            sample_weights.append(WEIGHT_ALIASED)
        else:
            sample_weights.append(WEIGHT_DEFAULT)

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # --- Load model ---
    print(f"  Loading model: {model_path}")
    model = torch.load(model_path, weights_only=False, mmap=False)
    if USE_MULTI_GPU:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Frozen reference model for KL constraint ---
    ref_model = torch.load(model_path, weights_only=False, mmap=False)
    if USE_MULTI_GPU:
        ref_model = nn.DataParallel(ref_model)
    ref_model = ref_model.to(DEVICE)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    baseline_reward = np.mean(rewards_all)
    target_row = STACK_SIZE - 1

    # --- Training loop ---
    print(f"  Starting Offline RL fine-tuning "
          f"(epochs={EPOCHS}, lr={LR})...")

    early_stopped = False
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for i, (images, rewards) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            rewards = rewards.to(DEVICE, non_blocking=True)

            logits = model(images)
            logits_target = logits[:, 0, target_row, :]
            target_bits = images[:, 0, target_row, :]

            # Log-probability under current policy (Bernoulli)
            m = torch.distributions.Bernoulli(logits=logits_target)
            log_probs = m.log_prob(target_bits).sum(dim=-1)

            # Entropy: -p*log(p) - (1-p)*log(1-p)
            probs = torch.sigmoid(logits_target)
            entropy = -(probs * torch.log(probs + 1e-8) +
                        (1 - probs) * torch.log(1 - probs + 1e-8)).sum(dim=-1)

            # Moving baseline
            batch_mean_reward = rewards.mean().item()
            baseline_reward = (BASELINE_MOMENTUM * baseline_reward +
                               (1 - BASELINE_MOMENTUM) * batch_mean_reward)

            # Clipped advantages
            advantages = rewards - baseline_reward
            advantages = torch.clamp(
                advantages, min=ADVANTAGE_CLIP_MIN, max=ADVANTAGE_CLIP_MAX)

            # Policy gradient loss
            pg_loss = -(log_probs * advantages).mean()

            # Full Bernoulli KL: p*log(p/q) + (1-p)*log((1-p)/(1-q))
            with torch.no_grad():
                ref_logits = ref_model(images)[:, 0, target_row, :]
            ref_probs = torch.sigmoid(ref_logits)
            curr_probs = probs
            kl_div = (curr_probs * torch.log(
                          curr_probs / (ref_probs + 1e-8) + 1e-8) +
                      (1 - curr_probs) * torch.log(
                          (1 - curr_probs) / (1 - ref_probs + 1e-8) + 1e-8)
                      ).sum(dim=-1)

            # Combined loss
            rl_loss = (pg_loss
                       - ENTROPY_COEF * entropy.mean()
                       + KL_COEF * kl_div.mean())

            optimizer.zero_grad()
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += rl_loss.item()

            if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
                print(f"    [Epoch {epoch+1}/{EPOCHS} | "
                      f"Batch {i+1}/{len(train_loader)}]: "
                      f"PG={pg_loss.item():.2f} | "
                      f"Ent={entropy.mean().item():.2f} | "
                      f"KL={kl_div.mean().item():.2f} | "
                      f"R={batch_mean_reward:.2f}")

            if entropy.mean().item() < ENTROPY_EARLY_STOP:
                print(f"    Early stop! Entropy {entropy.mean().item():.2f} "
                      f"< {ENTROPY_EARLY_STOP}")
                early_stopped = True
                break

        if early_stopped:
            break

    # --- Save fine-tuned model ---
    save_path = os.path.join(
        MODEL_DIR,
        f"cluster_{cl_idx}_{args.model_save_name}"
        f"_with_1_{STACK_SIZE}_{W}.pth")
    model_to_save = model.module if USE_MULTI_GPU else model
    torch.save(model_to_save, save_path)
    print(f"  Saved: {save_path}")

# ==========================================
# 7. Main: loop over all clusters
# ==========================================
if __name__ == "__main__":
    print(f">>> Offline RL retraining: k={K_VALUE}, stack_size={STACK_SIZE}")
    print(f"    Will process clusters 0 to {K_VALUE - 1}")
    print(f"    Model prefix: {args.model_name_prefix}")
    print(f"    Log prefix: {args.log_name_prefix}")
    print(f"    Save prefix: {args.model_save_name}")

    bgp_prefixes = load_bgp_prefixes(BGP_PREFIX_FILE)

    completed = []
    skipped = []

    for cl_idx in range(K_VALUE):
        try:
            retrain_cluster(cl_idx, bgp_prefixes)
            completed.append(cl_idx)
        except Exception as e:
            print(f"  Cluster {cl_idx} failed: {e}")
            skipped.append(cl_idx)

    print(f"\n{'=' * 60}")
    print(f">>> All done! k={K_VALUE}, stack_size={STACK_SIZE}")
    print(f"    Completed: {completed}")
    if skipped:
        print(f"    Skipped/Failed: {skipped}")
    print(f"{'=' * 60}")