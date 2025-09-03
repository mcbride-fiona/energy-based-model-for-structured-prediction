# utils.py
import torch
import string
from config import BETWEEN, DASH  # Import BETWEEN and DASH constants
from collections import Counter
import random

def transform_word(s):
    """
    Map a plaintext string (letters + '-') to indices with BETWEEN separators.
    Example: 'ab-c' -> [0, BETWEEN, 1, BETWEEN, DASH, BETWEEN, 2, BETWEEN]
    """
    result = []
    for ch in s:
        if ch == '-':  # dash token
            idx = DASH
        else:
            idx = ord(ch) - ord('a')
        result.append(idx)
        result.append(BETWEEN)
    return torch.tensor(result, dtype=torch.long)

def indices_to_str(indices):
    """
    Convert per-window argmin indices back to a plaintext string.
    Strategy: decode each window index to a char or separator token, then
    collapse windows between separators by taking the most frequent letter.
    """
    letters = list(string.ascii_lowercase)
    divisor = '_'       # internal BETWEEN marker for this function
    dash_char = '-'     # displayed dash

    idx_list = indices.cpu().tolist()
    char_list = []
    for i in idx_list:
        if i < 26:
            char_list.append(letters[i])
        elif i == BETWEEN:
            char_list.append(divisor)
        elif i == DASH:
            char_list.append(dash_char)

    # collapse runs between separators by majority vote
    result_chars, segment = [], []
    for char in char_list:
        if char != divisor:
            segment.append(char)
        else:
            if segment:
                counts = Counter(segment)
                most_common_char = counts.most_common(1)[0][0]
                result_chars.append(most_common_char)
                segment = []
    if segment:
        counts = Counter(segment)
        most_common_char = counts.most_common(1)[0][0]
        result_chars.append(most_common_char)

    return "".join(result_chars)

def simple_collate_fn(samples):
    images, annotations = zip(*samples)
    annotations = list(map(lambda c: torch.tensor(ord(c) - ord('a')), annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    images = [torch.nn.functional.pad(img, (0, m_width - img.shape[-1])) for img in images]
    return torch.stack(images), torch.stack(annotations)

def collate_fn(samples):
    """
    Return:
      images_t: [B, H, W]
      annotations_t: [B, T] (padded index sequences with BETWEEN)
      raw_texts: list[str] ground-truth plaintexts (for evaluation metrics)
    """
    images, annotations = zip(*samples)       # annotations are strings
    raw_texts = list(annotations)             # keep the ground-truth strings
    annotations = list(map(transform_word, annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    m_length = max(3, max([s.shape[0] for s in annotations]))
    images = [torch.nn.functional.pad(img, (0, m_width - img.shape[-1])) for img in images]
    annotations = [torch.nn.functional.pad(s, (0, m_length - s.shape[0]), value=BETWEEN) for s in annotations]
    return torch.stack(images), torch.stack(annotations), raw_texts
