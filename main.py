import torch
from model import SimpleNet
from data import SimpleWordsDataset
from utils import collate_fn, indices_to_str, transform_word
from train import train_ebm_model, build_ce_matrix, find_path, path_cross_entropy
from inference import plot_energies, plot_pm
import matplotlib.pyplot as plt
import string

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and dataloader
dataset = SimpleWordsDataset(2, len=2500, jitter=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Initialize model and optimizer
model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
train_ebm_model(model, 5, dataloader, optimizer)

# Visualize output
alphabet = dataset.draw_text(string.ascii_lowercase, 340).to(device)
alphabet = alphabet.unsqueeze(0).unsqueeze(0)
energies = model(alphabet)
targets = transform_word(string.ascii_lowercase).to(device)
pm = torch.gather(energies, 2, targets.unsqueeze(0).unsqueeze(1).expand(-1, energies.shape[1], -1))
free_energy, path, _, _ = find_path(build_ce_matrix(energies, targets.unsqueeze(0)).squeeze(0))

plot_pm(pm[0].detach().cpu(), path)
print("Free energy:", free_energy)

min_indices = energies[0].argmin(dim=-1)
print("Decoded:", indices_to_str(min_indices))
