from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .setup import Setup


class EyeTracking(object):

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Setup()

        # Detektor do wykrywania twarzy
        self._face_detector = dlib.get_frontal_face_detector()

        # Pobranie wyszkolonego modelu
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(
            cwd, "models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    # Sprawdza czy można było wykryć źrenice, True jeśli tak, False jeśli nie
    @property
    def pupils_located(self):
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    # Konwersja z BGR na szarość, wykrycie twarzy na obrazie, określenie punktów na twarzy
    def _analyze(self):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except:
            self.eye_left = None
            self.eye_right = None

    # Odświeżenie aktualnej klatki i jej analiza
    def refresh(self, frame):
        self.frame = frame
        self._analyze()

    # Zwraca wzpółrzędne lewej źrenicy
    def pupil_left_coords(self):
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    # Zwraca wzpółrzędne prawej źrenicy
    def pupil_right_coords(self):
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    # Zwraca stosunek położenia źrenicy od 0 do 1 w płaszczyźnie poziomej (0-zachód, 1-wschód)
    def horizontal_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / \
                (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / \
                (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    # Zwraca stosunek położenia źrenicy od 0 do 1 w płaszczyźnie pionowej (0-północ, 1-południe)
    def vertical_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / \
                (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / \
                (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2
