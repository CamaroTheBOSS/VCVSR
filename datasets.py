import os
from glob import glob
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from kornia.augmentation import ColorJiggle, RandomCrop, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, \
    Resize


class Vimeo90k(Dataset):
    def __init__(self, root: str, scale: int, test_mode: bool = False, crop_size: Tuple[int, int] = (256, 384)):
        super().__init__()
        self.root = root
        self.sequences = os.path.join(root, "train")
        self.txt_file = os.path.join(root, "sep_testlist.txt" if test_mode else "sep_trainlist.txt")

        self.scale = scale
        self.crop_size = crop_size
        self.test_mode = test_mode
        self.videos = self.load_paths()
        self.transform = Compose([ToTensor()])
        self.augment = Compose([
            # ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25), hue=(-0.02, 0.02),
            #             same_on_batch=True),
            RandomCrop(size=crop_size, same_on_batch=True),
            # RandomVerticalFlip(same_on_batch=True),
            # RandomHorizontalFlip(same_on_batch=True),
            # RandomRotation(degrees=180, same_on_batch=True),
        ])

        assert os.path.exists(self.root)
        assert os.path.exists(self.txt_file)

    def load_paths(self):
        videos = []
        with open(self.txt_file, "r") as f:
            for suffix in f.readlines():
                frame_paths = glob(os.path.join(self.sequences, suffix.strip(), "*.png"))
                if self.test_mode:
                    videos.append([path for path in frame_paths])
                else:
                    videos.extend([[frame_paths[i-1], frame_paths[i]] for i in range(1, len(frame_paths))])
        return videos

    def read_video(self, index):
        video = []
        for path in self.videos[index]:
            video.append(self.transform(Image.open(path).convert("RGB")))
        return torch.stack(video)

    def __getitem__(self, index: int):
        video = self.read_video(index)
        hqs = self.augment(video) if not self.test_mode else video[:, :, :self.crop_size[0], :self.crop_size[1]]
        n, c, h, w = hqs.shape
        lqs = Resize((int(h / self.scale), int(w / self.scale)))(hqs)

        return lqs, hqs

    def __len__(self) -> int:
        return len(self.videos)
