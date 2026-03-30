import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import ipaddress

# ==========================================
# 0. Command-line arguments
# ==========================================
parser = argparse.ArgumentParser(description="Cluster IPv6 seed addresses.")
parser.add_argument("--seed_file", type=str, required=True,
                    help="Path to your seed file.")
parser.add_argument("--label_file", type=str, default="./label.txt",
                    help="Output label file (default: ./label.txt).")
parser.add_argument("--k", type=int, default=6,
                    help="Number of clusters (default: 6).")
args = parser.parse_args()

# ==========================================
# 1. Configuration
# ==========================================
SEED_FILE = args.seed_file
LABEL_FILE = args.label_file

BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
LATENT_DIM = 32
N_CLUSTERS = args.k
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. Convolutional Autoencoder
# ==========================================
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder: (N, 1, 1, 128) -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 1 * 32, latent_dim),
        )

        # Decoder: latent_dim -> (N, 1, 1, 128)
        self.decoder_input = nn.Linear(latent_dim, 64 * 1 * 32)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 1, 32)),
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 3), stride=(1, 2),
                               padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(1, 3), stride=(1, 2),
                               padding=(0, 1), output_padding=(0, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(self.decoder_input(z))
        return x_recon, z

# ==========================================
# 3. Data preprocessing
# ==========================================
def expand_ipv6(addr):
    return ipaddress.IPv6Address(addr.strip()).exploded.replace(":", "")

def hex2bin_matrix(hex_str):
    """Hex string -> (1, 128) binary array."""
    bits = np.array([int(b) for b in bin(int(hex_str, 16))[2:].zfill(128)],
                    dtype=np.float32)
    return bits.reshape(1, 128)

print(">>> Loading IPv6 seeds...")
if not os.path.exists(SEED_FILE):
    raise FileNotFoundError(f"{SEED_FILE} not found!")

seeds = []
with open(SEED_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            seeds.append(line.strip())

print(f"Total seeds: {len(seeds)}")

print(">>> Encoding to binary matrices...")
data_list = []
valid_indices = []

for idx, addr in enumerate(seeds):
    try:
        img = hex2bin_matrix(expand_ipv6(addr))
        data_list.append(img)
        valid_indices.append(idx)
    except Exception:
        continue

X = np.array(data_list)         # (N, 1, 128)
X = np.expand_dims(X, axis=1)   # (N, 1, 1, 128)

tensor_x = torch.from_numpy(X).float()
dataset = TensorDataset(tensor_x)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. Train autoencoder
# ==========================================
model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f">>> Training Autoencoder on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for (x,) in train_loader:
        x = x.to(DEVICE)
        recon, z = model(x)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.6f}")

# ==========================================
# 5. Extract latent features
# ==========================================
print(">>> Extracting latent features...")
model.eval()
embeddings = []

with torch.no_grad():
    for (x,) in eval_loader:
        x = x.to(DEVICE)
        _, z = model(x)
        embeddings.append(z.cpu().numpy())

Z = np.concatenate(embeddings, axis=0)
print(f"Latent shape: {Z.shape}")

# ==========================================
# 6. Hierarchical clustering (landmark strategy for large datasets)
# ==========================================
print(">>> Normalizing features...")
Z = normalize(Z, norm='l2')

LANDMARK_SIZE = 20000
total_samples = Z.shape[0]

if total_samples > 30000:
    # Landmark strategy: cluster a subset, propagate via KNN
    print(f"\n[Info] Using Landmark Strategy (Ward linkage).")

    indices = np.random.permutation(total_samples)
    landmark_idx = indices[:LANDMARK_SIZE]
    remaining_idx = indices[LANDMARK_SIZE:]
    Z_landmark = Z[landmark_idx]

    print(">>> Running Agglomerative Clustering (Ward)...")
    hc = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, linkage='ward', metric='euclidean')
    landmark_labels = hc.fit_predict(Z_landmark)

    print(">>> Propagating labels via KNN...")
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(Z_landmark, landmark_labels)
    remaining_labels = knn.predict(Z[remaining_idx])

    final_labels = np.zeros(total_samples, dtype=int)
    final_labels[landmark_idx] = landmark_labels
    final_labels[remaining_idx] = remaining_labels

else:
    print(">>> Running Full Agglomerative Clustering (Ward)...")
    hc = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, linkage='ward', metric='euclidean')
    final_labels = hc.fit_predict(Z)

# ==========================================
# 7. Summary and save
# ==========================================
print("\nCluster Summary:")
unique_lbs, counts = np.unique(final_labels, return_counts=True)
for lb, count in zip(unique_lbs, counts):
    print(f"Cluster {lb}: {count} samples")

print(">>> Saving labels...")
with open(LABEL_FILE, "w", encoding="utf-8") as f:
    for lb in final_labels:
        f.write(str(lb) + "\n")

print(f"Done! Labels saved to {LABEL_FILE}")