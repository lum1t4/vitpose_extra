import os
import torch
import numpy as np
import cv2
from typing import Protocol
import logging


LOGGING_NAME = 'easyViTPose'
LOGGER = logging.getLogger(LOGGING_NAME)


HUGGINGFACE_REPO_ID = 'JunkyByte/easy_ViTPose'
SUPPORTED_DATASETS = ['coco', 'coco_25', 'wholebody', 'mpii', 'ap10k', 'apt36k', 'aic']
SUPPORTED_SIZES = ['s', 'b', 'l', 'h']
SUPPORTED_BACKENDS = ['torch', 'onnx', 'engine', 'coreml', 'torchscript']
BACKENDS_SUPPORTED_EXT = {
    "torch": ["pth", "torchscript", "pt"],
    "coreml": ["mlmodel", "mlpackage"],
    "onnx": ["onnx"],
    "tensorflow": ["tflite", "pb", "pbtxt"],
    "engine": ["trt", "engine", "plan"],
    "keras": ["h5", "hdf5"],
    "uff": ["uff"]
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs')
MODEL_CONFIGS = os.path.join(CONFIG_PATH, 'models')
USER_DIR = os.path.expanduser('~')
CACHE_DIR = os.path.join(USER_DIR, '.easyViTPose')

DETECTION_LABEL_2_IDS = {
    'human': [0],
    'cat': [15],
    'dog': [16],
    'horse': [17],
    'sheep': [18],
    'cow': [19],
    'elephant': [20],
    'bear': [21],
    'zebra': [22],
    'giraffe': [23],
    'animals': [15, 16, 17, 18, 19, 20, 21, 22, 23]
}

MODEL_ABBR_MAP = {
    's': 'small',
    'b': 'base',
    'l': 'large',
    'h': 'huge'
}


ROTATION_MAP = {
    0: None,
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE
}
