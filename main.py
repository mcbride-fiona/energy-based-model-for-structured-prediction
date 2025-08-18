import torch
import matplotlib.pyplot as plt
import string
import argparse
import os
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

from model import SimpleNet
from data import SimpleWordsDataset
from utils import collate_fn, indices_to_str, transform_word
from train import train_ebm_model, build_ce_matrix, find_path
from inference import plot_energies, plot_pm

try:
    from config import FONT_PATH
except ImportError:
    FONT_PATH = None

# === Parse command-line argument ===
parser = argparse.ArgumentParser()
parser.add_argument("--word", type=str, default=string.ascii_lowercase, help="Word to decode")
args = parser.parse_args()
word = args.word.lower()

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset and dataloader ===
dataset = SimpleWordsDataset(6, len=5000, jitter=True)  # Recommended: max_length=8, len=5000
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# === Initialize model and optimizer ===
model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Train model and save loss curve ===
loss_history = train_ebm_model(model, 10, dataloader, optimizer)

os.makedirs("outputs", exist_ok=True)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("EBM Training Loss Curve")
plt.savefig("outputs/training_loss_curve.png")
plt.close()

# === Save model weights ===
torch.save(model.state_dict(), "outputs/trained_model.pth")
print("Model saved to outputs/trained_model.pth")

# === Save ONLY the test word image ===
def save_text_image(text: str, path: str, length: Optional[int] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if length is None:
        length = 18 * len(text)

    img = Image.new('L', (length, 32), color=0)
    try:
        if FONT_PATH:
            fnt = ImageFont.truetype(FONT_PATH, 20)
        else:
            fnt = ImageFont.truetype("/Library/Fonts/Arial.ttf", 20)
    except Exception:
        fnt = ImageFont.load_default()

    d = ImageDraw.Draw(img)
    d.text((0, 5), text, fill=255, font=fnt)
    img.save(path)

safe_name = word.replace(' ', '_').replace('-', '_')
save_text_image(word, f"outputs/input_{safe_name}.png")

# === Run inference ===
img = dataset.draw_text(word).to(device)
img = img.unsqueeze(0).unsqueeze(0)
energies = model(img)
targets = transform_word(word).to(device)

pm = torch.gather(
    energies, 2,
    targets.unsqueeze(0).unsqueeze(1).expand(-1, energies.shape[1], -1)
)
free_energy, path, _, _ = find_path(
    build_ce_matrix(energies, targets.unsqueeze(0)).squeeze(0)
)

# === Save inference plots ===
plot_pm(pm[0].detach().cpu(), path, save_path=f"outputs/path_matrix_{safe_name}.png")
plot_energies(energies[0], save_path=f"outputs/energy_matrix_{safe_name}.png")

# === Print results ===
print("Free energy:", free_energy)
min_indices = energies[0].argmin(dim=-1)
print("Decoded:", indices_to_str(min_indices))
