import torch
import torch.nn.functional as F

def train_ebm_model(model, train_loader, optimizer):
    """
    Train for a SINGLE epoch (no internal tqdm).
    The outer loop in main.py controls total epochs and the progress bar.

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

        # Current training is per-sample; easy to vectorize later
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
            if j > i:
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
