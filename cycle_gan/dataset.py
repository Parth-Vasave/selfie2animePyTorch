# Copyright (c) 2018-2021, RangerUFO
#
# This file is part of cycle_gan.
#
# cycle_gan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cycle_gan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cycle_gan.  If not, see <https://www.gnu.org/licenses/>.


import os
import cv2
import random
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_dataset(name, category):
    url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/%s.zip" % (name)
    data_path = "data"
    if not os.path.exists(os.path.join(data_path, name)):
        print("Downloading dataset %s..." % name, flush=True)
        torch.hub.download_url_to_file(url, "%s.zip" % name)
        with zipfile.ZipFile("%s.zip" % name) as f:
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            f.extractall(path=data_path)
    imgs = [os.path.join(path, f) for path, _, files in os.walk(os.path.join(data_path, name, category)) for f in files]
    return imgs


def normalize(tensor):
    return (tensor - 0.5) / 0.5


def denormalize(tensor):
    return ((tensor * 0.5 + 0.5).clamp(0.0, 1.0) * 255).to(torch.uint8)


def reconstruct_color(img):
    return denormalize(img)


class CycleGANDataset(Dataset):
    def __init__(self, dataset_a, dataset_b, fine_size=(256, 256), load_size=(286, 286)):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.fine_size = fine_size
        self.load_size = load_size
        self.length = max(len(dataset_a), len(dataset_b))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_a = self._process(self.dataset_a[idx % len(self.dataset_a)])
        img_b = self._process(self.dataset_b[idx % len(self.dataset_b)])
        return img_a, img_b

    def _process(self, path):
        img = load_image(path)
        # Random rotation [-20, 20] degrees
        h, w = img.shape[:2]
        angle = random.uniform(-20, 20)
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(img, mat, (w, h), flags=random.randint(0, 4))
        # Resize short side to load_size
        short = min(h, w)
        scale = min(self.load_size) / short
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=random.randint(0, 4))
        # Random crop to fine_size
        crop_h, crop_w = self.fine_size
        top = random.randint(0, max(0, new_h - crop_h))
        left = random.randint(0, max(0, new_w - crop_w))
        img = img[top:top + crop_h, left:left + crop_w]
        # Random horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
        # Random color distort
        img = Image.fromarray(img)
        img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(img)
        # To tensor and normalize to [-1, 1]
        img = transforms.ToTensor()(img)
        img = normalize(img)
        return img


def get_batches(dataset_a, dataset_b, batch_size, fine_size=(256, 256), load_size=(286, 286), device=torch.device("cpu")):
    dataset = CycleGANDataset(dataset_a, dataset_b, fine_size, load_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    for batch_a, batch_b in loader:
        yield batch_a.to(device), batch_b.to(device)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    batch_size = 4
    dataset_a = load_dataset("vangogh2photo", "trainA")
    dataset_b = load_dataset("vangogh2photo", "trainB")
    for batch_a, batch_b in get_batches(dataset_a, dataset_b, batch_size):
        print("batch_a preview: ", batch_a.shape)
        print("batch_b preview: ", batch_b.shape)
        for i in range(batch_size):
            plt.subplot(batch_size * 2 // 8 + 1, 4, i + 1)
            plt.imshow(reconstruct_color(batch_a[i].permute(1, 2, 0)).numpy())
            plt.axis("off")
            plt.subplot(batch_size * 2 // 8 + 1, 4, i + batch_size + 1)
            plt.imshow(reconstruct_color(batch_b[i].permute(1, 2, 0)).numpy())
            plt.axis("off")
        plt.show()
