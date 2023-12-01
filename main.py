import os
import re
from typing import Any
from huggingface_hub import hf_hub_download
from constants import HUGGINGFACE_REPO_ID, LOGGER, BACKENDS_SUPPORTED_EXT, MODEL_CONFIGS
import cv2
import yaml
from typing import Callable, Tuple, List, Dict, Optional
import numpy as np

from nn import ViTPose
from ops.keypoints import keypoints_from_heatmaps
from entity import PoseDataset

from constants import (
    SUPPORTED_DATASETS, SUPPORTED_SIZES, SUPPORTED_BACKENDS,
    BACKENDS_SUPPORTED_EXT, MODEL_CONFIGS, HUGGINGFACE_REPO_ID,
    CONFIG_PATH, CACHE_DIR, MODEL_ABBR_MAP
)


def parse_model_info(
        path: str,
        dataset: str = None,
        size: str = None,
        backend: str = "torch"
    ):
    mapping = {v: k for k, v in BACKENDS_SUPPORTED_EXT.items() for v in v}
    expr = re.compile(r"ViTPose[-_]([sblh])[-_](\w+)", re.IGNORECASE)

    dir_path = os.path.dirname(path)
    default_model = len(dir_path) == 0
    path = os.path.realpath(path)
    filename = os.path.basename(path)
    filename, ext = os.path.splitext(filename)
    ext = ext[1:]  # remove dot
    filename = filename.lower()
    backend = mapping.get(ext)
    match = expr.match(filename)

    if match:
        size, dataset = match.groups()
    else:
        raise ValueError(
            f"Please specify the dataset and model size " \
                "in the filename or in the arguments."
        )
    
    if default_model:
        return get_default_path(dataset, size, backend) 
    return path, backend, size, dataset


def get_default_path(dataset, size, backend):
    if backend == "torch":
        remote_path = \
            os.path.join("torch", f'{dataset}/vitpose-{size}-{dataset}.pth')
        path = hf_hub_download(HUGGINGFACE_REPO_ID, remote_path)
    elif backend == "onnx":
        remote_path = \
            os.path.join("onnx", f'{dataset}/vitpose-{size}-{dataset}.onnx')
        path = hf_hub_download(HUGGINGFACE_REPO_ID, remote_path)
    elif backend == "coreml":
        path = os.path.join(CACHE_DIR, f"ViTPose-{size}-{dataset}.mlpackage")
    elif backend == "tensorrt":
        path = os.path.join(CACHE_DIR, f"ViTPose-{size}-{dataset}.trt")
    else:
        raise ValueError(f"Backend {backend} not supported.")
    
    return path, backend, size, dataset


def check_local(path):
    dir_path = os.path.dirname(path)
    dir_path = None if len(dir_path) == 0 else dir_path
    filename = os.path.basename(path)
    filename = filename.lower()
    return dir_path, filename


def pad_image(image: np.ndarray, aspect_ratio: float) -> np.ndarray:
    # Get the current aspect ratio of the image
    image_height, image_width = image.shape[:2]
    current_aspect_ratio = image_width / image_height

    left_pad = 0
    top_pad = 0
    # Determine whether to pad horizontally or vertically
    if current_aspect_ratio < aspect_ratio:
        # Pad horizontally
        target_width = int(aspect_ratio * image_height)
        pad_width = target_width - image_width
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        padded_image = np.pad(image,
                              pad_width=((0, 0), (left_pad, right_pad), (0, 0)),
                              mode='constant')
    else:
        # Pad vertically
        target_height = int(image_width / aspect_ratio)
        pad_height = target_height - image_height
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad

        padded_image = np.pad(image,
                              pad_width=((top_pad, bottom_pad), (0, 0), (0, 0)),
                              mode='constant')
    return padded_image, (left_pad, top_pad)


class PoseModel:
    def __init__(self, wieghts, config, dataset: PoseDataset) -> None:
        self.weights = wieghts
        self.config = config
        self.dataset = dataset

    def __call__(self, im: np.ndarray) -> Any:
        B = im.shape[0]
        return np.random.randn(B, *self.dataset.keypoints_shape) \
            .astype(np.float32)


def prepocess(
    img: np.ndarray,
    bbox: np.ndarray,
    image_size: tuple | list | np.ndarray = (192, 256),
    mean: tuple | list | np.ndarray = [0.485, 0.456, 0.406],
    std: tuple | list | np.ndarray = [0.229, 0.224, 0.225]
) -> tuple[np.ndarray, int, int, int, int]:
    bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-10, 10], 0, img.shape[1])
    bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-10, 10], 0, img.shape[0])
    # Crop image and pad to 3/4 aspect ratio
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    img, (left_pad, top_pad) = pad_image(img, 3 / 4)
    org_h, org_w = img.shape[:2]
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR) / 255
    img = np.expand_dims(((img - mean) / std).transpose(2, 0, 1), axis=0) \
        .astype(np.float32)
    return img, org_h, org_w, left_pad, top_pad


def predict(
        model: Callable[[np.ndarray], np.ndarray],
        img: np.ndarray,
        bbox: np.ndarray,
        **kargs
    ) -> np.ndarray:
    cropped_img, org_h, org_w, left_pad, top_pad = prepocess(img, bbox, **kargs)
    heatmaps = model(cropped_img)
    keypoints = np.squeeze(postprocess_keypoints(heatmaps, org_w, org_h))
    keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
    return keypoints


def postprocess_keypoints(heatmaps, org_w, org_h) -> np.ndarray:
    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps,
        center=np.array([[org_w // 2, org_h // 2]]),
        scale=np.array([[org_w, org_h]]),
        unbiased=True,
        use_udp=True
    )
    return np.concatenate([points[:, :, ::-1], prob], axis=2)


def load_backend(backend, path, config):
    if backend == "torch":
        from backends.torch import TorchBackend
        return TorchBackend(config).load(path)
    elif backend == "onnx":
        pass
        # from .backends.onnx import ONNXBackend
        # return ONNXBackend().load(path)
    elif backend == "tensorrt":
        from backends.tensorrt import TensorRTBackend
        return TensorRTBackend().load(path)
    elif backend == "coreml":
        from backends.coreml import CoreMLBackend
        return CoreMLBackend().load(path)
    else:
        raise ValueError(f"Backend {backend} not supported.")


class ViTPoseAutoBackend:
    def __init__(
        self,
        path: str = "ViTPose-s-coco.pt",
        config: str = os.path.join(MODEL_CONFIGS, 'vitpose.yaml'),
        dataset: str = None,
        size: str = None
    ):
        path, backend, size, dataset = parse_model_info(path, dataset, size)

        assert dataset in SUPPORTED_DATASETS, \
            f"Dataset {dataset} not supported. " \
            f"Supported datasets are {SUPPORTED_DATASETS}"
        
        assert size in SUPPORTED_SIZES, \
            f"Model size {size} not supported. " \
            f"Supported sizes are {SUPPORTED_SIZES}"
        
        assert backend in SUPPORTED_BACKENDS, \
            f"Backend {self.backend} not supported. " \
            f"Supported backends are {SUPPORTED_BACKENDS}"
        
        assert os.path.isfile(config) and os.path.exists(config), \
            f"Config file {config} does not exist."
        
        
        self.dataset = PoseDataset() \
            .load(os.path.join(CONFIG_PATH, 'datasets', f'{dataset}.yaml'))
        
        with open(config, "r") as f:
            loaded = yaml.safe_load(f)
            
            model_config = loaded[f"model_{MODEL_ABBR_MAP[size]}"]
            model_config["keypoint_head"]["out_channels"] = \
                self.dataset.keypoints_shape[0]
            self.data_config = loaded["data_config"]

        self.model = load_backend(backend, path, model_config)

    def __str__(self):
        import json
        model_config = json.dumps(self.model_config, indent=4)
        data_config = json.dumps(self.data_config, indent=4)
        return f"Model config: {model_config}\n" \
            f"Data config: {data_config}\n" \
            f"Dataset: {self.dataset}\n"

    def __call__(self, img: np.ndarray, bboxes: np.ndarray, stream: bool = False, **kwds: Any) -> Any:
        kargs = {**self.data_config, **kwds}
        image_size = kargs["image_size"] if "image_size" in kargs \
            else self.data_config["image_size"]
        mean = kargs["mean"] if "mean" in kargs \
            else self.data_config["normalization"]["mean"]
        std = kargs["std"] if "std" in kargs \
            else self.data_config["normalization"]["std"]
        
        return [predict(self.model , img, bbox, image_size=image_size,
                        mean=mean, std=std) for bbox in bboxes]
    
    def get_dataset(self):
        return self.dataset