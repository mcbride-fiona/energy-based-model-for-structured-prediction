import torch
from model import SimpleNet
from data import SimpleWordsDataset
from utils import collate_fn, indices_to_str, transform_word
from train import train_ebm_model, build_ce_matrix, find_path, path_cross_entropy
from inference import plot_energies, plot_pm
import matplotlib.pyplot as plt
import string
import argparse
import os

# === Parse command-line argument ===
parser = argparse.ArgumentParser()
parser.add_argument("--word", type=str, default=string.ascii_lowercase, help="Word to decode")
args = parser.parse_args()
word = args.word.lower()

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset and dataloader ===
dataset = SimpleWordsDataset(2, len=2500, jitter=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# === Initialize model and optimizer ===
model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Train model ===
train_ebm_model(model, 5, dataloader, optimizer)

# === Run inference on custom word ===
img = dataset.draw_text(word).to(device)
img = img.unsqueeze(0).unsqueeze(0)
energies = model(img)
targets = transform_word(word).to(device)

pm = torch.gather(energies, 2, targets.unsqueeze(0).unsqueeze(1).expand(-1, energies.shape[1], -1))
free_energy, path, _, _ = find_path(build_ce_matrix(energies, targets.unsqueeze(0)).squeeze(0))

# === Save plots ===
os.makedirs("outputs", exist_ok=True)
plot_pm(pm[0].detach().cpu(), path, save_path=f"outputs/path_matrix_{word}.png")
plot_energies(energies[0], save_path=f"outputs/energy_matrix_{word}.png")

# === Print results ===
print("Free energy:", free_energy)

min_indices = energies[0].argmin(dim=-1)
print("Decoded:", indices_to_str(min_indices))

