from __future__ import division
import cv2
from .pupil import Pupil


class Setup(object):

    def __init__(self):
        self.nb_frames = 30
        self.thresholds_left = []
        self.thresholds_right = []

    # Sprawdzenie ukończenia kalibracji
    def is_complete(self):
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    # Obliczenie progu
    def threshold(self, side):
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    # Zajętość przestrzenna tęczówki w oku
    @staticmethod
    def iris_size(frame):
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    # Określenie najlepszego progu
    def evaluate(self, eye_frame, side):
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Setup.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(
            lambda p: abs(p[1] - average_iris_size)))
        threshold = best_threshold

        # Dla danego oka
        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)
