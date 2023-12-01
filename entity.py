from typing import Tuple, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt


def color_palette(color_palette='tab20', palette_samples=16):


    try:
        palette = np.round(np.array(
            plt.get_cmap(color_palette).colors
        ) * 255).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        palette = np.round(np.array(
            plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))
        ) * 255).astype(np.uint8)[:, -2::-1].tolist()
    return palette


class PoseDataset:
    def __init__(self):
        self.keypoints_shape = (17, 3)
        self.keypoints: Dict[int, str] = None
        self.joints = None
        self.reverse_mapping = None
    
    def load(self, path: str) -> Self:
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        self.keypoints_shape = data['keypoints_shape']
        self.keypoints = data['keypoints']
        self.joints = data['skeleton']

        self.palette_joints = color_palette(
            color_palette="tab20",
            palette_samples=self.keypoints_shape[0]
        )
        self.palette_edges = color_palette(
            color_palette='Set2',
            palette_samples=len(self.joints)
        )

        self.reverse_mapping = {}
        for idx, name in self.keypoints.items():
            self.reverse_mapping[name] = idx
        
        
        return self
    
    def kpt_color(self, idx: int):
        return tuple(self.palette_joints[idx % len(self.palette_joints)])
    
    def joint_color(self, idx: int):
        return tuple(self.palette_edges[idx % len(self.palette_edges)])

    def __repr__(self):
      return f"Keypoints shape: {self.keypoints_shape},\n" \
            f"Keypoints' names: {list(self.reverse_mapping.keys())}\n" \
            f"Skeleton: {self.joints}"


class Plotter:
    def __init__(self, dataset: PoseDataset) -> None:
        self.dataset = dataset

    def plot(self, img, bboxes: np.ndarray | list = None, keypoints: np.ndarray | list = None, **kwds: Any) -> Any:

        show_bbox = kwds.get("show_bbox", True)
        show_keypoints = kwds.get("show_keypoints", True)
        
        annotated = img.copy()
        for bbox, kps in zip(bboxes, keypoints):
            if bbox is not None and show_bbox:
                annotated = draw_bbox(annotated, bbox, "person")
            if kps is not None and show_keypoints:
                annotated = self.draw_skeleton(annotated, kps)
        return annotated
    
    
    def draw_skeleton(self, img, keypoints, conf=0.5):
        for i, k in enumerate(keypoints):
            draw_keypoint(img, k, self.dataset.kpt_color(i), conf)

        for i, joint in enumerate(self.dataset.joints):
            draw_joint(img, keypoints, joint, self.dataset.joint_color(i), conf=conf)

        return img



def draw_bbox(img: np.ndarray, bbox: np.ndarray, cls: str):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(img, cls, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def draw_keypoint(
        img: np.ndarray,
        keypoint: np.ndarray,
        color: tuple = (255, 255, 255),
        conf: float = 0.5
    ):

    radius = max(1, min(img.shape[:2]) // 150, 6)
    if (len(keypoint.shape) > 2 and keypoint[2] > conf) or len(keypoint.shape) == 2:
        img = cv2.circle(img, keypoint[[1, 0]], radius, color, -1)


def draw_joint(
        img: np.ndarray,
        keypoints: np.ndarray,
        joint: np.ndarray | tuple,
        color: tuple = (255, 255, 255),
        conf: float = 0.5
    ):
    k1, k2 = keypoints[joint]
    if k1[2] > conf and k2[2] > conf:
        img = cv2.line(img, k1[[1, 0]].astype(np.int32), k2[[1, 0]].astype(np.int32), color, 2)
        # pt1 = (int(k1[1]), int(k1[0]))
        # pt2 = (int(k2[1]), int(k2[0]))
        # img = cv2.line(img, pt1, pt2, color, 2)
