"""
Data loading and preprocessing utils.
"""

# ==================== [IMPORT] ====================

import cv2
import torch
import numpy as np
import pytorch_lightning as pl

import os
import json
import math
import random
import pathlib
from PIL import Image, ImageDraw


# ==================== [IMPORT] ====================

TRAIN_DATASET_ROOT = './data/train/'
VALID_DATASET_ROOT = './data/valid/'
TEST_DATASET_ROOT = './data/test/'


# ===================== [CODE] =====================


def generate_random_mask(height: int = 256,
                         width: int = 256,
                         min_lines: int = 1,
                         max_lines: int = 4,
                         min_vertex: int = 5,
                         max_vertex: int = 13,
                         mean_angle: float = 2/5 * math.pi,
                         angle_range: float = 2/15 * math.pi,
                         min_width: float = 12,
                         max_width: float = 40):
    """
    Generate random mask for GAN. Each pixel of mask
    if 1 or 0, 1 means pixel is masked.

    Parameters
    ----------
    height : int
        Height of mask.
    width : int
        Width of mask.
    min_lines : int
        Miniumal count of lines to draw on mask.
    max_lines : int
        Maximal count of lines to draw on mask.
    min_vertex : int
        Minimal count of vertexes to draw.
    max_vertex : int
        Maximum count of vertexes to draw.
    mean_angle : float
        Mean value of angle between edges.
    angle_range : float
        Maximum absoulte deviation of angle from mean value.
    min_width : int
        Minimal width of edge to draw.
    max_width : int
        Maximum width of edge to draw.
    """

    # init mask and drawing tool
    mask = Image.new('1', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # calculate mean radius to draw lines and count of lines
    num_lines = np.random.randint(min_lines, max_lines)
    average_radius = math.sqrt(height * height + width * width) / 8

    # drawing lines
    for _ in range(num_lines):
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        num_vertex = np.random.randint(min_vertex, max_vertex)

        # line parameters
        angles = []
        vertex = []

        # generating line angles
        for i in range(num_vertex - 1):
            random_angle = np.random.uniform(angle_min, angle_max)
            if i % 2 == 0:
                random_angle = 2 * np.pi - random_angle
            angles.append(random_angle)

        # start point
        start_x = np.random.randint(0, width)
        start_y = np.random.randint(0, height)
        vertex.append((start_x, start_y))

        # generating next points
        for i in range(num_vertex - 1):
            radius = np.random.normal(loc=average_radius, scale=average_radius / 2)
            radius = np.clip(radius, 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + radius * math.cos(angles[i]), 0, width)
            new_y = np.clip(vertex[-1][1] + radius * math.sin(angles[i]), 0, height)
            vertex.append((int(new_x), int(new_y)))

        # drawing line
        line_width = np.random.uniform(min_width, max_width)
        line_width = int(line_width)
        draw.line(vertex, fill=1, width=line_width)

        # smoothing angles
        for node in vertex:
            x_ul = node[0] - line_width // 2
            x_br = node[0] + line_width // 2
            y_ul = node[1] - line_width // 2
            y_br = node[1] + line_width // 2
            draw.ellipse((x_ul, y_ul, x_br, y_br), fill=1)

    # random vertical flip
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)

    # random horizontal flip
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)

    mask = np.asarray(mask, np.float32)
    return torch.from_numpy(mask)



class Dataset(torch.utils.data.Dataset):

    def __init__(self, root: str,
                       file_ext: str = 'jpg',
                       seed: int = 42,
                       max_size: int = None,
                       create_index: bool = True):
        """
        Dataset constructor.

        Parameters
        ----------
        root : str
            Root directory of dataset.
        make_mask : bool
            If true, generates mask for each image.
        file_ext : str
            Extention of files to look for in root directory.
        seed : int
            Seed for random shuffle.
        """

        super(Dataset, self).__init__()

        self.root = root
        self.file_ext = file_ext
        self.seed = seed
        self.max_size = max_size

        index_filename = os.path.join(self.root, 'index.json')

        if os.path.exists(index_filename):
            with open(index_filename, 'r') as file:
                self.files = json.load(file)
        else:
            root_path = pathlib.Path(self.root)
            self.files = list(root_path.rglob('*.' + self.file_ext))
            self.files = [str(x) for x in self.files]

        if create_index:
            with open(index_filename, 'w') as file:
                json.dump(self.files, file)

        # shuffling
        random.seed(self.seed)
        random.shuffle(self.files)


    def __getitem__(self, index: int) -> (torch.Tensor, str):
        """
        Get image and its filename by key.

        Parameters
        ----------
        index : int
            Index of item to fetch.
        """

        filename = self.files[index]
        image = cv2.imread(filename).astype(np.float)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        return image


    def __len__(self):
        """
        Get dataset size.
        """

        if self.max_size is not None:
            return min(len(self.files), self.max_size)

        return len(self.files)



class MaskedDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super(MaskedDataset, self).__init__(*args, **kwargs)


    def __getitem__(self, index: int) -> (torch.Tensor, str, torch.Tensor):
        """
        Get image, image filename and mask.

        Parameters
        ----------
        index : int
            Index of item to return.
        """

        image = super().__getitem__(index)
        mask = torch.from_numpy(generate_random_mask())
        return image, mask



class PlacesDataModule(pl.LightningDataModule):

    def __init__(self, train_root: str = TRAIN_DATASET_ROOT,
                       valid_root: str = VALID_DATASET_ROOT,
                       test_root: str = TEST_DATASET_ROOT,
                       batch_size: int = 4,
                       num_workers: int = 16):
        """
        Pytorch-ligtning datamodule for places2 dataset.

        Parameters
        ----------
        train_root : str
            Root directory for train dataset.
        valid_root : str
            Root directory for test dataset.
        test_root : str
            Root directory for test dataset.
        batch_size : int
            Batch size.
        num_workers: int
            Number of dataloader workers.
        """

        super(PlacesDataModule, self).__init__()

        self.train_root = train_root
        self.valid_root = valid_root
        self.test_root = test_root
        self.batch_size = batch_size
        self.num_workers = num_workers


    def train_dataloader(self):
        dataset = MaskedDataset(self.train_root)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


    def val_dataloader(self):
        dataset = MaskedDataset(self.valid_root, max_size=10)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


    def test_dataloader(self):
        dataset = MaskedDataset(self.test_root)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

