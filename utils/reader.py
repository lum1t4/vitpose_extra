from constants import ROTATION_MAP
import cv2
from datetime import timedelta


class VideoReader(object):
    def __init__(self, file_name, rotate=0):
        self.file_name = file_name
        self.rotate = ROTATION_MAP[rotate]
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        print(
            f"fps: {self.fps}, frame_count: {self.frame_count}, "
            f"duration: {timedelta(seconds=self.duration)}"
        )

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration()
        if self.rotate is not None:
            img = cv2.rotate(img, self.rotate)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def __len__(self):
        return self.frame_count