import os
import argparse
import string
from typing import Optional

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from model import SimpleNet
from data import SimpleWordsDataset
from utils import collate_fn, indices_to_str, transform_word
from train import train_ebm_model, build_ce_matrix, find_path
from inference import plot_energies, plot_pm

try:
    from config import FONT_PATH
except Exception:
    FONT_PATH = None

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--word", type=str, default=string.ascii_lowercase, help="Word to decode")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--skip-train", action="store_true", help="Skip training and load checkpoint if available")
parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pth", help="Checkpoint path")
args = parser.parse_args()
word = args.word.lower()

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Datasets & loaders
# ----------------------------
train_ds = SimpleWordsDataset(6, len=5000, jitter=True)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn)

# small deterministic validation sample set (no jitter)
val_ds = SimpleWordsDataset(6, len=32, jitter=False)
val_examples = ["hello", "protein", "image", "label"]  # tweak as you like

# ----------------------------
# Model / Optimizer
# ----------------------------
model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ----------------------------
# Output dirs
# ----------------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("outputs/val_examples", exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def save_text_image(text: str, path: str, length: Optional[int] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if length is None:
        length = 18 * len(text)
    img = Image.new('L', (length, 32), color=0)  # black bg
    try:
        if FONT_PATH:
            fnt = ImageFont.truetype(FONT_PATH, 20)
        else:
            fnt = ImageFont.truetype("/Library/Fonts/Arial.ttf", 20)
    except Exception:
        fnt = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    d.text((0, 5), text, fill=255, font=fnt)  # white text
    img.save(path)

@torch.no_grad()
def evaluate_and_save(model, epoch: int):
    """Evaluate on a few fixed texts; save qualitative artifacts; return simple char accuracy."""
    model.eval()
    total_hits, total_chars = 0, 0
    for text in val_examples:
        # Save raw rendered image
        img_t = val_ds.draw_text(text)  # [H,W] tensor in [0,1]
        from torchvision.transforms.functional import to_pil_image
        to_pil_image(img_t).save(f"outputs/val_examples/epoch_{epoch:04d}_image_{text}.png")

        # Forward pass
        ein = img_t.unsqueeze(0).unsqueeze(0).to(device)
        energies = model(ein)

        # Decode (argmin over classes)
        min_idx = energies[0].argmin(dim=-1)
        pred = indices_to_str(min_idx)

        # Simple char-accuracy
        m = min(len(pred), len(text))
        total_hits += sum(int(pred[i] == text[i]) for i in range(m))
        total_chars += len(text)

        # Save energy + path plots
        tgt = transform_word(text).to(device)
        pm = torch.gather(energies, 2, tgt.unsqueeze(0).unsqueeze(1).expand(-1, energies.shape[1], -1))
        fe, path, _, _ = find_path(build_ce_matrix(energies, tgt.unsqueeze(0)).squeeze(0))
        plot_pm(pm[0].detach().cpu(), path, save_path=f"outputs/val_examples/epoch_{epoch:04d}_path_{text}.png")
        plot_energies(energies[0], save_path=f"outputs/val_examples/epoch_{epoch:04d}_energy_{text}.png")
    model.train()
    return (total_hits / total_chars) if total_chars else 0.0

# ----------------------------
# Train loop (epoch-by-epoch) OR load
# ----------------------------
loss_history = []
best_acc = -1.0

if args.skip_train and os.path.exists(args.checkpoint):
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
else:
    for epoch in range(1, args.epochs + 1):
        # Run exactly ONE epoch so we can log each epoch
        avg_train_loss = train_ebm_model(model, 1, train_loader, optimizer)
        loss_history.append(avg_train_loss)

        # Save/update loss curve
        plt.figure(dpi=140)
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("EBM Training Loss")
        plt.tight_layout()
        plt.savefig("outputs/training_loss_curve.png")
        plt.close()

        # Evaluate + save validation artifacts
        char_acc = evaluate_and_save(model, epoch)

        # Save best checkpoint (by char accuracy)
        if char_acc >= best_acc:
            best_acc = char_acc
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved best checkpoint to {args.checkpoint} (char_acc={best_acc:.3f})")

# ----------------------------
# Inference on the provided word
# ----------------------------
safe_name = word.replace(' ', '_').replace('-', '_')
save_text_image(word, f"outputs/input_{safe_name}.png")

img = val_ds.draw_text(word).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    energies = model(img)
    targets = transform_word(word).to(device)

pm = torch.gather(
    energies, 2,
    targets.unsqueeze(0).unsqueeze(1).expand(-1, energies.shape[1], -1)
)
free_energy, path, _, _ = find_path(
    build_ce_matrix(energies, targets.unsqueeze(0)).squeeze(0)
)

plot_pm(pm[0].detach().cpu(), path, save_path=f"outputs/path_matrix_{safe_name}.png")
plot_energies(energies[0], save_path=f"outputs/energy_matrix_{safe_name}.png")

min_indices = energies[0].argmin(dim=-1)
decoded = indices_to_str(min_indices)
print("Free energy:", float(free_energy))
print("Decoded:", decoded)
