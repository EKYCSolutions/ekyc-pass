import numpy as np
import pandas as pd
import cv2
from .tracker_bytetrack import TrackerBytetrack


class Track():
    def __init__(self, model_type="byetrack") -> None:
        self.model_type = model_type
        self.tracker = None

    def build(self):
        if self.model_type == "byetrack":
            self.tracker = TrackerBytetrack()
            

    def process(self):
        self.tracker.track()
