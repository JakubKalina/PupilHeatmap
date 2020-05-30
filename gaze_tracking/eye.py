import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):

    # Punkty lewego oka z modelu
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]

    # Punkty prawego oka z modelu
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None

        self._analyze(original_frame, landmarks, side, calibration)

    # Zwraca punkt po środku pomiedzy dwoma podanymi punktami
    @staticmethod
    def _middle_point(p1, p2):
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    # Odizolowanie oka od reszty obrazu
    def _isolate(self, frame, landmarks, points):
        region = np.array(
            [(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)

        # Nakładanie maski aby wyodrębnić oko
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        # Ustawienie punktu środkowego obrazu
        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    # Izolacja oka z obrazu, dokonanie kalibracji, nowy obiekt źrenicy
    def _analyze(self, original_frame, landmarks, side, calibration):

        # W zależności czy prawe czy lewe oko
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
