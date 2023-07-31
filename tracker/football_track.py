import numpy as np
import pandas as pd

import cv2

from .football_track_bytetrack import Football_bytetrack


class FootballTrack():
    def __init__(self, model_type="byetrack") -> None:
        self.model_type = model_type
        self.tracker = None

    def build(self):
        if self.model_type == "byetrack":
            self.tracker = Football_bytetrack()

    def process(self):
        self.tracker.track()
