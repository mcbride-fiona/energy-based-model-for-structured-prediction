import os
import argparse
import string
from typing import Optional, List

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from model import SimpleNet
from data import SimpleWordsDataset
from utils import collate_fn, transform_word
from train import train_ebm_model, build_ce_matrix, find_path
from inference import plot_energies, plot_pm
from metrics_ebm_text import RunningStats
from train_logger import TrainLogger, save_config

try:
    # Need BETWEEN and DASH for decoding
    from config import FONT_PATH, BETWEEN, DASH
except Exception:
    FONT_PATH, BETWEEN, DASH = None, None, None


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

# fixed validation set (no jitter/noise) for stable metrics
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
        # a tiny bit more generous than 18*len to avoid cramping the tail char
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


def decode_segmented(energies_1LC: torch.Tensor) -> str:
    """
    DP-based segmentation decoder.

    Steps:
      1) Dynamic program over states {LETTER, BETWEEN} across windows to find an
         optimal segmentation (monotonic, like the training alignment).
      2) For each contiguous LETTER segment, pick the letter/dash whose SUM energy
         over that segment is minimal.

    This is far more stable than per-window argmin + majority vote, especially for
    trailing characters.

    Args:
      energies_1LC: [L, C] tensor for one image

    Returns:
      plaintext str (letters + '-')
    """
    assert BETWEEN is not None and DASH is not None, "BETWEEN/DASH must be defined in config.py"
    L, C = energies_1LC.shape

    LETTER_CLASSES = list(range(26)) + [DASH]  # allow letters + '-'
    # cost per window for 'being a LETTER window' vs 'BETWEEN window'
    letter_cost_per_win, _ = energies_1LC[:, LETTER_CLASSES].min(dim=1)  # [L]
    between_cost_per_win = energies_1LC[:, BETWEEN]                      # [L]

    # DP over states s ∈ {0=LETTER, 1=BETWEEN}
    dp = torch.full((L, 2), float('inf'), device=energies_1LC.device)
    prev_state = torch.full((L, 2), -1, dtype=torch.long, device=energies_1LC.device)

    # Start in LETTER (labels start with a letter then BETWEEN)
    dp[0, 0] = letter_cost_per_win[0]   # LETTER
    dp[0, 1] = float('inf')             # disallow BETWEEN at start

    for i in range(1, L):
        # LETTER state
        stay_L = dp[i-1, 0] + letter_cost_per_win[i]
        switch_BL = dp[i-1, 1] + letter_cost_per_win[i]
        if stay_L <= switch_BL:
            dp[i, 0] = stay_L; prev_state[i, 0] = 0
        else:
            dp[i, 0] = switch_BL; prev_state[i, 0] = 1

        # BETWEEN state
        stay_B = dp[i-1, 1] + between_cost_per_win[i]
        switch_LB = dp[i-1, 0] + between_cost_per_win[i]
        if stay_B <= switch_LB:
            dp[i, 1] = stay_B; prev_state[i, 1] = 1
        else:
            dp[i, 1] = switch_LB; prev_state[i, 1] = 0

    # Prefer ending in BETWEEN (labels end with BETWEEN)
    end_state = 1 if dp[L-1, 1] <= dp[L-1, 0] else 0

    # Backtrack states to get segmentation
    states = [end_state]; i = L - 1; s = end_state
    while i > 0:
        s_prev = prev_state[i, s].item()
        states.append(s_prev)
        s = s_prev; i -= 1
    states.reverse()  # len L, entries in {0,1}

    # For each LETTER segment, pick the class minimizing SUM energy over the segment
    out_chars = []
    i = 0
    while i < L:
        if states[i] == 0:  # LETTER segment
            j = i
            while j + 1 < L and states[j + 1] == 0:
                j += 1
            segE = energies_1LC[i:j+1, LETTER_CLASSES].sum(dim=0)  # [27]
            best = LETTER_CLASSES[int(segE.argmin().item())]
            if best == DASH:
                out_chars.append('-')
            else:
                out_chars.append(chr(ord('a') + best))
            i = j + 1
        else:
            i += 1

    return "".join(out_chars)


@torch.no_grad()
def free_energy_of_batch(energies: torch.Tensor, target_idxs: torch.Tensor) -> List[float]:
    """
    Compute free energy for each sample in the batch by building the CE matrix and
    running the DP path finder.
      energies: [B, L, C]
      target_idxs: [B, T]
    returns: list[float] of length B
    """
    B = energies.size(0)
    fes = []
    for b in range(B):
        ce = build_ce_matrix(energies[b:b+1], target_idxs[b:b+1]).squeeze(0)  # [L, T]
        fe, _path, _dp, _diag = find_path(ce)
        fes.append(float(fe))
    return fes


@torch.no_grad()
def evaluate(model) -> dict:
    """
    Validation loop:
      - Decode via DP-based segmentation (decode_segmented)
      - Compute E(gold) via DP on gold targets
      - Compute E(pred) via DP on targets produced by decoded string
      - Accumulate task and energy metrics with RunningStats
    """
    model.eval()
    stats = RunningStats()

    for images, targets, texts in val_loader:
        images = images.unsqueeze(1).to(device).float()   # [B,1,H,W]
        targets = targets.to(device)                      # [B,T]
        energies = model(images)                          # [B,L,C]

        # Decode each sample with DP-segmentation
        preds = [decode_segmented(energies[b]) for b in range(energies.size(0))]

        # E(gold): free energy computed with gold indices
        E_gold_list = free_energy_of_batch(energies, targets)

        # E(pred): convert decoded strings to indices, pad to common T, then DP
        pred_targets_list = [transform_word(p) for p in preds]
        T_max = max(t.shape[0] for t in pred_targets_list)
        pred_targets = torch.stack([
            torch.nn.functional.pad(t, (0, T_max - t.shape[0]), value=BETWEEN)
            for t in pred_targets_list
        ]).to(device)
        E_pred_list = free_energy_of_batch(energies, pred_targets)

        # Update metrics
        for pred, gold_text, e_g, e_p in zip(preds, texts, E_gold_list, E_pred_list):
            stats.update(pred=pred, gold=gold_text, e_gold=e_g, e_pred=e_p)

    return stats.summary()


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
best_acc = -1.0

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

        # update progress bar postfix so you can see the loss live
        epoch_iter.set_postfix({"train_loss": f"{avg_train_loss:.3f}"})

        # Save/update loss curve
        plt.figure(dpi=140)
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("EBM Training Loss")
        plt.tight_layout()
        plt.savefig("outputs/training_loss_curve.png")
        plt.close()

        # --- evaluate on val set, log metrics ---
        metrics = evaluate(model)
        print(f"[Epoch {epoch}/{args.epochs}] " +
              ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in metrics.items()))

        logger.add(
            epoch=epoch,
            train_loss=avg_train_loss,
            cer=metrics.get("CER_mean"),
            char_acc=metrics.get("CharAcc"),
            fe_mean=metrics.get("E_gold_mean"),
            fe_std=None,  # could add np.std over E_gold if desired
        )
        logger.flush_csv()
        logger.plot_loss()

        # Save best checkpoint by character accuracy (proxy task metric)
        cur_acc = metrics.get("CharAcc", 0.0)
        if cur_acc >= best_acc:
            best_acc = cur_acc
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved best checkpoint to {args.checkpoint} (char_acc={best_acc:.3f})")


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

# NEW: decode with DP-based segmentation (more robust than per-window argmin)
decoded = decode_segmented(energies[0])
print("Free energy:", float(free_energy))
print("Decoded:", decoded)
