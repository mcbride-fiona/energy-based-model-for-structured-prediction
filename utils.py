import torch
from config import BETWEEN, DASH  # Needed for target encoding

def transform_word(s: str) -> torch.Tensor:
    """
    Map a plaintext string (letters + '-') to indices with BETWEEN separators.
    Example: 'ab-c' -> [0, BETWEEN, 1, BETWEEN, DASH, BETWEEN, 2, BETWEEN]
    """
    out = []
    for ch in s:
        if ch == '-':
            idx = DASH
        else:
            idx = ord(ch) - ord('a')  # assumes lowercase a-z
        out.append(idx)
        out.append(BETWEEN)
    return torch.tensor(out, dtype=torch.long)

def collate_fn(samples):
    """
    Collate a list of (image_tensor[H,W], text:str) into:
      - images_t:   [B, H, W]   (padded to same width)
      - targets_t:  [B, T]      (BETWEEN-separated, padded to max T with BETWEEN)
      - raw_texts:  list[str]   (ground-truth strings for evaluation)
    """
    images, texts = zip(*samples)  # texts are strings
    raw_texts = list(texts)

    # encode targets with BETWEEN separators
    targets = [transform_word(t) for t in raw_texts]

    # pad images to same width
    max_w = max(18, max(img.shape[-1] for img in images))
    images_padded = [torch.nn.functional.pad(img, (0, max_w - img.shape[-1])) for img in images]

    # pad targets to same length with BETWEEN
    max_t = max(3, max(t.shape[0] for t in targets))
    targets_padded = [torch.nn.functional.pad(t, (0, max_t - t.shape[0]), value=BETWEEN) for t in targets]

    return torch.stack(images_padded), torch.stack(targets_padded), raw_texts
