import string, random
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from config import FONT_PATH

class SimpleWordsDataset(IterableDataset):
    def __init__(self, max_length, len=100, jitter=False, noise=False):
        self.max_length = max_length
        self.len = len
        self.jitter = jitter
        self.noise = noise
        self.transforms = transforms.ToTensor()

    def __len__(self):
        return self.len

    def __iter__(self):
        for _ in range(self.len):
            text = ''.join(random.choices(string.ascii_lowercase, k=self.max_length))
            yield self.draw_text(text), text

    def draw_text(self, text, length=None):
        if length is None:
            length = 18 * len(text)
        img = Image.new('L', (length, 32))
        fnt = ImageFont.truetype(FONT_PATH, 20)
        d = ImageDraw.Draw(img)
        pos = (random.randint(0, 7), 5) if self.jitter else (0, 5)
        d.text(pos, text, fill=1, font=fnt)
        img = self.transforms(img)
        img[img > 0] = 1
        if self.noise:
            img += torch.bernoulli(torch.ones_like(img) * 0.1)
            img.clamp_(0, 1)
        return img[0]
