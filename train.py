# train.py
import os
import csv
import json
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Tokens for BETWEEN and DASH used in decoding/encoding
from config import BETWEEN, DASH


# ============================================================
# Decoding helper (DP-based segmentation)
# ============================================================

def decode_segmented(energies_1LC: torch.Tensor) -> str:
    """
    DP-based segmentation decoder.

    Steps:
      1) Dynamic program over states {LETTER, BETWEEN} across windows to find an
         optimal segmentation (monotonic, like the training alignment).
      2) For each contiguous LETTER segment, pick the letter/dash whose SUM energy
         over that segment is minimal.

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


# ============================================================
# Training and alignment utilities
# ============================================================

def train_ebm_model(model, train_loader, optimizer):
    """
    Train for a SINGLE epoch (no internal tqdm).
    The outer loop controls how many epochs we run.

    Args:
        model: torch.nn.Module
        train_loader: yields (images [B,H,W], targets [B,T], raw_texts list[str])
        optimizer: torch optimizer

    Returns:
        avg_epoch_loss (float)
    """
    device = next(model.parameters()).device
    model.train()

    epoch_loss = 0.0
    sample_count = 0

    for images, targets, _texts in train_loader:
        batch_size = images.size(0)

        # Current training is per-sample; easy to vectorize later if needed
        for i in range(batch_size):
            optimizer.zero_grad()

            image = images[i].unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,H,W]
            target = targets[i].to(device)                                  # [T]

            energies = model(image)    # [1,L,C]
            ce = build_ce_matrix(energies, target.unsqueeze(0)).squeeze(0)  # [L,T]

            free_energy, path, _, _ = find_path(ce)

            loss = path_cross_entropy(ce, path)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            sample_count += 1

    return epoch_loss / max(sample_count, 1)


def build_ce_matrix(energies, targets):
    """
    energies: [B, L, C] raw energies per window and class
    targets:  [B, T]    target class indices (BETWEEN-separated sequence)
    Returns:  [B, L, T] per-(window, target-position) cross-entropy matrix.
    """
    B, L, C = energies.shape
    T = targets.shape[1]
    log_probs = F.log_softmax(-energies, dim=-1)         # convert energies to log-probs
    log_probs_exp = log_probs.unsqueeze(2).expand(B, L, T, C)
    index = targets.unsqueeze(1).expand(B, L, T).unsqueeze(-1)
    gathered_log_probs = log_probs_exp.gather(dim=3, index=index).squeeze(-1)
    return -gathered_log_probs  # cross-entropy per (L,T)


def find_path(pm):
    """
    Dynamic programming to find the lowest-cost monotonic path through pm[L,T].
    Moves: stay (advance in L), or diagonal (advance in L and T).
    Returns (free_energy, path, dp, diag).
    """
    L, T = pm.shape
    dp = pm.new_full((L, T), float('inf'))
    diag = torch.zeros((L, T), dtype=torch.bool, device=pm.device)

    dp[0, 0] = pm[0, 0]
    for j in range(1, T):
        dp[0, j] = float('inf')
    for i in range(1, L):
        dp[i, 0] = dp[i-1, 0] + pm[i, 0]
    for i in range(1, L):
        for j in range(1, T):
            if j > i:  # optional speedup: maintain monotonicity constraint
                continue
            stay = dp[i-1, j] + pm[i, j]
            move = dp[i-1, j-1] + pm[i, j]
            if move < stay:
                dp[i, j] = move
                diag[i, j] = True
            else:
                dp[i, j] = stay

    if dp[L-1, T-1] == float('inf'):
        return float('inf'), [], dp, diag

    path = []
    i, j = L - 1, T - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0 and j > 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if diag[i, j]:
                i, j = i-1, j-1
            else:
                i -= 1
    path.append((0, 0))
    path.reverse()
    return dp[L-1, T-1].item(), path, dp, diag


def path_cross_entropy(ce, path):
    """
    Accumulate cross-entropy along the chosen path.
    """
    total = ce[0, 0] * 0.0
    for l, t in path:
        total += ce[l, t]
    return total


# ============================================================
# Evaluation helper (char accuracy)
# ============================================================

@torch.no_grad()
def evaluate_char_acc(model, data_loader, device: Optional[torch.device] = None) -> float:
    """
    Evaluate character accuracy on a given data loader using decode_segmented.
    The collate_fn must return (images, targets, raw_texts).
    """
    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    total_hits, total_chars = 0, 0
    for images, _targets, texts in data_loader:
        images = images.unsqueeze(1).to(device).float()   # [B,1,H,W]
        energies = model(images)                          # [B,L,C]
        for b in range(energies.size(0)):
            pred = decode_segmented(energies[b])
            gold = texts[b]
            m = min(len(pred), len(gold))
            total_hits += sum(1 for i in range(m) if pred[i] == gold[i])
            total_chars += len(gold)

    if was_training:
        model.train()
    return total_hits / max(total_chars, 1)


# ============================================================
# Lightweight Logger + Config Snapshot
# ============================================================

class TrainLogger:
    """
    Minimal CSV logger (no pandas dependency) plus a simple loss plot.
    Call:
        logger = TrainLogger(out_dir="outputs")
        logger.add(epoch, train_loss, char_acc=0.98)
        logger.flush_csv()
        logger.plot_loss()
    """
    def __init__(self, out_dir="outputs"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self.rows = []
        self.header_written = os.path.exists(self.csv_path)

    def add(self, epoch, train_loss, val_loss=None, cer=None, wer=None, char_acc=None, fe_mean=None, fe_std=None):
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "cer": cer,
            "wer": wer,
            "char_acc": char_acc,
            "free_energy_mean": fe_mean,
            "free_energy_std": fe_std,
        }
        self.rows.append(row)

    def flush_csv(self):
        if not self.rows:
            return
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerows(self.rows)
        self.rows.clear()

    def plot_loss(self):
        """
        Plot train/val loss from the CSV (no pandas required).
        If you never logged val_loss, it'll just plot train_loss.
        """
        if not os.path.exists(self.csv_path):
            return

        epochs, train_losses, val_losses = [], [], []
        has_val = False
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row["epoch"]))
                    train_losses.append(float(row["train_loss"]) if row["train_loss"] != "" else math.nan)
                    if "val_loss" in row and row["val_loss"] not in (None, "", "nan"):
                        val_losses.append(float(row["val_loss"]))
                        has_val = True
                    else:
                        val_losses.append(math.nan)
                except Exception:
                    continue

        if not epochs:
            return

        plt.figure(dpi=140)
        plt.plot(epochs, train_losses, label="train")
        if has_val:
            plt.plot([e for e, v in zip(epochs, val_losses) if not math.isnan(v)],
                     [v for v in val_losses if not math.isnan(v)],
                     label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "loss_curve.png"))
        plt.close()


def save_config(cfg: dict, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
