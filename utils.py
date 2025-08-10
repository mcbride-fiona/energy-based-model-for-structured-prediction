import torch
import string
from config import BETWEEN, DASH  # Import the new SPACE constant
from collections import Counter
import random

def transform_word(s):
    result = []
    for ch in s:
        if ch == '-':  # Check for a dash
            idx = DASH
        else:
            idx = ord(ch) - ord('a')
        result.append(idx)
        result.append(BETWEEN)
    return torch.tensor(result, dtype=torch.long)

def indices_to_str(indices):
    letters = list(string.ascii_lowercase)
    divisor = '_'
    dash_char = '-' # Use a dash for the decoded character

    idx_list = indices.cpu().tolist()
    char_list = []
    for i in idx_list:
        if i < 26:
            char_list.append(letters[i])
        elif i == BETWEEN:
            char_list.append(divisor)
        elif i == DASH: # Check for the DASH index
            char_list.append(dash_char)

    result_chars, segment = [], []
    for char in char_list:
        if char != divisor:
            segment.append(char)
        else:  # We've hit a separator
            if segment:
                counts = Counter(segment)
                if counts:
                    # Append the most common character in the segment
                    most_common_char = counts.most_common(1)[0][0]
                    result_chars.append(most_common_char)
                segment = []
    
    # Process the final segment after the loop ends
    if segment:
        counts = Counter(segment)
        if counts:
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
    images, annotations = zip(*samples)
    annotations = list(map(transform_word, annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    m_length = max(3, max([s.shape[0] for s in annotations]))
    images = [torch.nn.functional.pad(img, (0, m_width - img.shape[-1])) for img in images]
    annotations = [torch.nn.functional.pad(s, (0, m_length - s.shape[0]), value=BETWEEN) for s in annotations]
    return torch.stack(images), torch.stack(annotations)
