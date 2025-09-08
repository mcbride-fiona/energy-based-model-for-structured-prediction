import os
import argparse
import string
from typing import Optional

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from model import SimpleNet
from data import SimpleWordsDataset
from utils import collate_fn, transform_word
from train import (
    train_ebm_model, build_ce_matrix, find_path,
    TrainLogger, save_config, evaluate_char_acc, decode_segmented
)
from inference import plot_energies, plot_pm

try:
    # Need FONT_PATH for rendering
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
train_ds = SimpleWordsDataset(6, len=5000, jitter=True, noise=False)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn)

# fixed validation set (no jitter/noise) for stable character-accuracy
val_ds = SimpleWordsDataset(6, len=256, jitter=False, noise=False)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)


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
    """
    Render a plaintext string to a PNG (for qualitative inspection).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if length is None:
        # a bit wider than 18*len to avoid cramping the tail char
        length = 20 * len(text)
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


# ----------------------------
# Logger / config snapshot
# ----------------------------
logger = TrainLogger("outputs")
save_config({
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lr": args.lr,
})


# ----------------------------
# Train for N epochs with a single, clear progress bar
# ----------------------------
loss_history = []
best_char_acc = -1.0

if args.skip_train and os.path.exists(args.checkpoint):
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
else:
    epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        # ONE epoch of training
        avg_train_loss = train_ebm_model(model, train_loader, optimizer)
        loss_history.append(avg_train_loss)

        # Validation: character accuracy (uses DP-based decoder)
        char_acc = evaluate_char_acc(model, val_loader, device=device)

        # update progress bar postfix so you can see metrics live
        epoch_iter.set_postfix({"train_loss": f"{avg_train_loss:.3f}",
                                "char_acc": f"{char_acc:.3f}"})

        # Save/update loss curve
        plt.figure(dpi=140)
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", label="train loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("EBM Training")
        plt.tight_layout()
        plt.savefig("outputs/training_loss_curve.png")
        plt.close()

        # Log (train loss + char accuracy)
        logger.add(
            epoch=epoch,
            train_loss=avg_train_loss,
            char_acc=char_acc,
        )
        logger.flush_csv()
        logger.plot_loss()

        # Save best checkpoint by character accuracy
        if char_acc >= best_char_acc:
            best_char_acc = char_acc
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved best checkpoint to {args.checkpoint} (char_acc={best_char_acc:.3f})")


# ----------------------------
# Inference on the provided word (qualitative)
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

# Decode with DP-based segmentation (more robust than per-window argmin)
decoded = decode_segmented(energies[0])
print("Free energy:", float(free_energy))
print("Decoded:", decoded)
