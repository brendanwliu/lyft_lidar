import os
import gc
import numpy as np
import pandas as pd

import json
import math
import sys
import time
from datetime import datetime
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
<<<<<<< HEAD
from sklearn.preprocessing import LabelEncoder
=======
>>>>>>> cb6b869e2e493ef0b436209cc5cd3cef3c977e19
from PIL import Image

from matplotlib.axes import Axes
from matplotlib import animation, rc
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import plot, init_notebook_mode
import plotly.figure_factory as ff

import seaborn as sns
from pyquaternion import Quaternion
from tqdm import tqdm

from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from pathlib import Path
<<<<<<< HEAD
from keras.utils import to_categorical
=======

>>>>>>> cb6b869e2e493ef0b436209cc5cd3cef3c977e19
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
import copy

<<<<<<< HEAD
input_path = '/media/brendanliu/1ffe4965-0a76-4845-aedc-1929d41a1cde/lyft_vis/3d-object-detection-for-autonomous-vehicles/'
# train_csv = pd.read_csv(input_path+'train.csv')
# sample_submission = pd.read_csv(input_path+'sample_submission.csv')
# #Using the kaggle challenge description of the data.
# column_names = ['sample_token', 'object_id', 'center_x', 'center_y',
#                     'center_z', 'width', 'length', 'height', 'yaw','class_name']
=======
input_path = '../lyft_vis/3d-object-detection-for-autonomous-vehicles/'
train_csv = pd.read_csv(input_path+'train.csv')
sample_submission = pd.read_csv(input_path+'sample_submission.csv')
#Using the kaggle challenge description of the data.
column_names = ['sample_id', 'object_id', 'center_x', 'center_y',
                    'center_z', 'width', 'length', 'height', 'yaw','class_name']
>>>>>>> cb6b869e2e493ef0b436209cc5cd3cef3c977e19
# objects = []
# for sample_id, values in tqdm(train_csv.values[:]):
#     data_params = values.split()
#     num_obj = len(data_params)
#     for i in range(num_obj // 8):
#         x, y, z, w, l, h, yaw, c = tuple(data_params[i * 8: (i + 1) * 8])
#         objects.append([sample_id,i,x,y,z,w,l,h,yaw,c])
# train_data = pd.DataFrame(objects,columns=column_names)

# numerical_cols = ['object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 
#                     'height', 'yaw']
# train_data[numerical_cols] = np.float32(train_data[numerical_cols].values)
# train_data.to_csv(input_path+'train_dataframe.csv')

<<<<<<< HEAD
# def render_scene(index):
#     my_scene = lyft_dataset.scene[index]
#     my_sample_token = my_scene["first_sample_token"]
#     lyft_dataset.render_sample(my_sample_token)


traindf = pd.read_csv(input_path + 'train_dataframe.csv')
testdf = pd.read_csv(input_path + 'sample_submission.csv')

shape = (100, 100, 3)
MAX_VALUE = 140

train_data = pd.read_csv(input_path + 'train.csv')

lyft_data = LyftDataset(data_path = input_path, json_path = input_path + 'train_data')

categories = [i['name'] for i in lyft_data.category]

columns = ['confidence' ,'center_x', 'center_y', "center_z", 'width', 'length', 'height', 'rotate_w', 'rotate_x', 'rotate_y', 'rotate_z', 'class']
sensors = lyft_data.sensor
sensors = [i['channel'] for i in sensors]
sensors = [i for i in sensors if 'LIDAR' not in i]

def getImageFileNames(token : str):
    
    list_of_filenames = []
    
    for sensor in sensors:
        filename = lyft_data.get('sample_data', lyft_data.get('sample', token)['data'][sensor])['filename']
        filename = input_path + 'train_images' + filename[6:]
        list_of_filenames.append(filename)
        
    return list_of_filenames 

def getData(token):
    
    list_of_values = []
    list_of_anns = lyft_data.get('sample', token)['anns']
    for annotation_token in list_of_anns:
        sample_data = lyft_data.get('sample_annotation', annotation_token)
        list_of_values.append(sample_data['category_name'])
    
    return np.array(list_of_values)

print(len(getImageFileNames('8567b06acde454482d5577fee49918902ddb3218dfad09d7205cfd201258e304')))
print(len(getData('8567b06acde454482d5577fee49918902ddb3218dfad09d7205cfd201258e304')))

# def one_hot_encoding(value):
#     global categories
    
#     x = categories.index(value)
    
#     return [0] * (x) + [1] + [0] * (len(categories) - x)


# for token in tqdm(traindf['sample_token']):
#     allfiles = []
#     values = getData(token)
#     print(values)
#     # values = values.reshape((values.shape[0],) + (1, ) + (values.shape[1], ))
#     # filenames = getImageFileNames(token)
#     # allfiles.append(filenames)
#     # images = [np.asarray(Image.open(i).resize(shape[:-1])).reshape(shape) for i in filenames]

# traindf['filename'] = [item for sublist in allfiles for item in sublist]
# traindf.to_csv(input_path+'trainfile_dataframe.csv')
# load yolov3 model
# from keras.models import load_model
# model = load_model('/media/brendanliu/1ffe4965-0a76-4845-aedc-1929d41a1cde/lyft_vis/' + 'yolov3_model.h5')
# model.compile(loss='mean_squared_error', optimizer='sgd')
# model_fit(model)
=======
lyft_dataset = LyftDataset(data_path = input_path, json_path = input_path + 'train_data')
my_scene = lyft_dataset.scene[0]
print(my_scene)
>>>>>>> cb6b869e2e493ef0b436209cc5cd3cef3c977e19
