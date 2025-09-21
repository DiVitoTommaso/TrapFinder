# -------------------------------
# Standard library
# -------------------------------
import os
import json
import random
import math
import copy
import io

# -------------------------------
# Scientific / numerical libraries
# -------------------------------
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold

# -------------------------------
# PyTorch core
# -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.nn import MSELoss, L1Loss
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from geomloss import SamplesLoss
# -------------------------------
# PyTorch / torchvision transforms
# -------------------------------
import torchvision.transforms.functional as TF
from torchvision import transforms, models
from PIL import Image, ImageDraw
import timm

# -------------------------------
# Utilities
# -------------------------------
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_POINTS = 7

INPUT_SIZE = (224, int(224 * 4 / 3))  # H x W (keep aspect ratio 4:3)
ROOT_DIR = "Data"
LABEL = "Air-Bomb"