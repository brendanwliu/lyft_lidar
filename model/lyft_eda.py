import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
from PIL import Image

import seaborn as sns
from pyquaternion import Quaternion
from tqdm import tqdm

from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from pathlib import Path

import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
import copy

input_path = './3d-object-detection-for-autonomous-vehicles/'
train_data = pd.read_csv(input_path + 'train_dataframe.csv')

fig,ax = plt.subplots(figsize=(10,10))
sns.distplot(train_data['center_y'], color='purple', ax = ax)
sns.distplot(train_data['center_x'], ax = x)
plt.show()