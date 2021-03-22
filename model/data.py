"""
Data loading and preprocessing utils.
"""

# ==================== [IMPORT] ====================

import cv2
import torch
import numpy as np

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
    return mask



class PlacesDataset(torch.utils.data.Dataset):

    def __init__(self, root: str,
                       make_mask: bool = True,
                       file_ext: str = 'jpg',
                       seed: int = 42):
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

        super(PlacesDataset, self).__init__()

        self.root = root
        self.file_ext = file_ext
        self.seed = seed
        self.make_mask = make_mask

        # loading file names
        root_path = pathlib.Path(self.root)
        self.files = list(root_path.rglob('*.' + self.file_ext))
        self.files = [str(x) for x in self.files]

        # shuffling
        random.seed(self.seed)
        random.shuffle(self.files)


    def __getitem__(self, index: int):
        """
        Get dataset element by key.

        Parameters
        ----------
        index : int
            Index of item to fetch.
        """

        filename = self.files[index]
        image = cv2.imread(filename).astype(np.float)
        image = np.transpose(image, axes=(2, 0, 1))

        if self.make_mask:
            mask = generate_random_mask(width=256, height=256)
            mask = mask.reshape(1, 256, 256)
            image = np.concatenate([image, mask], axis=0)

        image = torch.from_numpy(image)
        return image


    def __len__(self):
        """
        Get dataset size.
        """

        return len(self.files)

