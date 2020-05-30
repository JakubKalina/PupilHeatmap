import cv2
from gaze_tracking.eyeTracking import EyeTracking
from array import *
import matplotlib.pyplot as plt

eye = EyeTracking()
webcam = cv2.VideoCapture(0)

# Wyzerowanie licznika
recordedHeatmap = [[0, 0],
                   [0, 0]]

# Jeden z wymiarów heatmapy
heatmapResolution = 2


while True:
    # Pobranie z kamery nowej klatki
    _, frame = webcam.read()

    # Analiza aktualnej klatki
    eye.refresh(frame)

    # Zaznaczenie wykrytych źrenic
    if eye.pupils_located:
        x_left, y_left = eye.pupil_left_coords()
        x_right, y_right = eye.pupil_right_coords()
        cv2.line(frame, (x_left - 5, y_left),
                 (x_left + 5, y_left), (0, 0, 255))
        cv2.line(frame, (x_left, y_left - 5),
                 (x_left, y_left + 5), (0, 0, 255))
        cv2.line(frame, (x_right - 5, y_right),
                 (x_right + 5, y_right), (0, 0, 255))
        cv2.line(frame, (x_right, y_right - 5),
                 (x_right, y_right + 5), (0, 0, 255))

        # Północny zachód
        if eye.horizontal_ratio() <= 0.5 and eye.vertical_ratio() <= 0.5:
            recordedHeatmap[0][0] += 1

        # Północny wschód
        if eye.horizontal_ratio() >= 0.5 and eye.vertical_ratio() <= 0.5:
            recordedHeatmap[0][1] += 1

        # Południowy wschód
        if eye.horizontal_ratio() >= 0.5 and eye.vertical_ratio() >= 0.5:
            recordedHeatmap[1][1] += 1

        # Południowy zachód
        if eye.horizontal_ratio() <= 0.5 and eye.vertical_ratio() >= 0.5:
            recordedHeatmap[1][0] += 1

    cv2.imshow("Pupil heatmap", frame)

    # Wyjście z programu klawiszem ESC
    if cv2.waitKey(1) == 27:
        print(recordedHeatmap)
        plt.imshow(recordedHeatmap)
        plt.show()
        break
