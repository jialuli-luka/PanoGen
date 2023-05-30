from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
from dataset.utils import pre_caption

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


decord.bridge.set_bridge("torch")


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, img):
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class vln_dataset(Dataset):
    def __init__(self, ann_file, data_root, max_words=30, read_local_data=True, is_train=True, max_img_size=384):

        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        # self.ann = self.ann[:100]

        self.max_words = max_words
        self.read_local_data = read_local_data

        self.max_img_size = max_img_size
        self.data_root = data_root
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.preprocess = Compose([Resize(max_img_size, interpolation=BICUBIC),
                                    CenterCrop(max_img_size),
                                    _convert_image_to_rgb,
                                    ToTensor(),
                                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                    ])
        self.is_train = is_train

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        path = ann['path']
        instruction_id = ann['instruction_id']
        imgs = []
        for vp in path:
            if not os.path.exists(os.path.join(self.data_root, vp)):
                img = self.preprocess(Image.open(os.path.join("views_img_sd", vp)))
            else:
                img = self.preprocess(Image.open(os.path.join(self.data_root, vp)))
            imgs.append(img)

        path = torch.stack(imgs)

        if self.is_train:
            instructions = ann['instruction']
            return path, instructions
        else:
            return path, instruction_id

class vln_rvr_dataset(Dataset):
    def __init__(self, ann_file, data_root, max_words=30, read_local_data=True, is_train=True, max_img_size=384):

        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        # self.ann = self.ann[:100]

        self.max_words = max_words
        self.read_local_data = read_local_data

        self.max_img_size = max_img_size
        self.data_root = data_root
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.preprocess = Compose([Resize(max_img_size, interpolation=BICUBIC),
                                    CenterCrop(max_img_size),
                                    _convert_image_to_rgb,
                                    ToTensor(),
                                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                    ])
        self.is_train = is_train

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        path = ann['path']
        instruction_id = ann['instruction_id']
        prompt = ann['obj']
        imgs = []
        for vp in path:
            if not os.path.exists(os.path.join(self.data_root, vp)):
                img = self.preprocess(Image.open(os.path.join("views_img_sd", vp)))
            else:
                img = self.preprocess(Image.open(os.path.join(self.data_root, vp)))
            imgs.append(img)

        path = torch.stack(imgs)

        if self.is_train:
            instructions = ann['instruction']
            return path, instructions, prompt
        else:
            return path, instruction_id, prompt
