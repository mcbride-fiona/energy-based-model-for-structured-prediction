import torch
import string
from config import BETWEEN
from collections import Counter
import random

def transform_word(s):
    result = []
    for ch in s:
        idx = ord(ch) - ord('a')
        result.append(idx)
        result.append(BETWEEN)
    return torch.tensor(result, dtype=torch.long)

def simple_collate_fn(samples):
    images, annotations = zip(*samples)
    annotations = list(map(lambda c: torch.tensor(ord(c) - ord('a')), annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    images = [torch.nn.functional.pad(img, (0, m_width - img.shape[-1])) for img in images]
    return torch.stack(images), torch.stack(annotations)

def collate_fn(samples):
    images, annotations = zip(*samples)
    annotations = list(map(transform_word, annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    m_length = max(3, max([s.shape[0] for s in annotations]))
    images = [torch.nn.functional.pad(img, (0, m_width - img.shape[-1])) for img in images]
    annotations = [torch.nn.functional.pad(s, (0, m_length - s.shape[0]), value=BETWEEN) for s in annotations]
    return torch.stack(images), torch.stack(annotations)

def indices_to_str(indices):
    letters = list(string.ascii_lowercase)
    divisor = '_'
    idx_list = indices.cpu().tolist()
    char_list = [letters[i] if i < 26 else divisor for i in idx_list]
    result_chars, segment = [], []
    for char in char_list:
        if char != divisor:
            segment.append(char)
        else:
            if segment:
                counts = Counter(segment)
                max_count = counts.most_common(1)[0][1]
                candidates = [ch for ch, cnt in counts.items() if cnt == max_count]
                result_chars.append(random.choice(candidates))
                segment = []
            result_chars.append(divisor)
    if segment:
        counts = Counter(segment)
        max_count = counts.most_common(1)[0][1]
        candidates = [ch for ch, cnt in counts.items() if cnt == max_count]
        result_chars.append(random.choice(candidates))
    return ''.join(result_chars).replace(divisor, '')
