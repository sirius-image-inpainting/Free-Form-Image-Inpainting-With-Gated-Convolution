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
                         min_edges: int = 4,
                         max_edges: int = 12,
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
    min_edges : int
        Minimal count of edges to draw.
    max_edges : int
        Maximum count of edges to draw.
    mean_angle : float
        Mean value of angle between edges.
    angle_range : float
        Maximum absoulte deviation of angle from mean value.
    min_width : int
        Minimal width of edge to draw.
    max_width : int
        Maximum width of edge to draw.
    """

    average_radius = math.sqrt(height * height + width * width) / 8
    mask = Image.new('L', (width, height), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_edges, max_edges)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)

    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, height, width, 1))
    return mask



class PlacesDataset(torch.utils.data.Dataset):

    def __init__(self, root: str,
                       transform: bool = True,
                       make_mask: bool = True,
                       file_ext: str = 'jpg',
                       seed: int = 42):
        """
        Dataset constructor.

        Parameters
        ----------
        root : str
            Root directory of dataset.
        transform : bool
            Transform image or not.
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
        self.transform = transform
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
        image = cv2.imread(filename).astype(numpy.float)

        if self.transform:
            image = (image / 255) * 2 - 1

        if self.make_mask:
            mask = generate_random_mask(width=256, height=256)
            image = numpy.concatenate([image, mask], axis=1)

        image = torch.from_numpy(image)
        return image


    def __len__(self):
        """
        Get dataset size.
        """

        return len(self.files)

