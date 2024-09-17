import cv2  # type: ignore
from utils import get_parking_spots_bboxes, empty_or_not
import numpy as np


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


video_path = "parking_1920_1080.mp4"
mask = "mask_1920_1080.png"

mask = cv2.imread(mask, 0)

cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)
frame_num = 0
spots_status_list = [None for j in spots]
diffs = [None for j in spots]
ret = True
previous_frame = None
step = 30
while ret:
    ret, frame = cap.read()

    if frame_num % step == 0 and previous_frame is not None:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1 : y1 + h, x1 : x1 + w, :]
            diffs[spot_index] = calc_diff(
                spot_crop, previous_frame[y1 : y1 + h, x1 : x1 + w, :]
            )

    if frame_num % step == 0:
        if previous_frame is None:
            arr = range(len(spots))
        else:
            arr = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_index in arr:
            x1, y1, w, h = spots[spot_index]
            spot_crop = frame[y1 : y1 + h, x1 : x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status_list[spot_index] = spot_status
        previous_frame = frame.copy()
    counter = 0
    for spot_index, spot in enumerate(spots):

        x1, y1, w, h = spots[spot_index]
        if spots_status_list[spot_index]:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            counter += 1
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
    frame_num += 1
    cv2.rectangle(frame, (80, 20), (555, 80), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "Available spots : {} / {}".format(str(counter), str(len(spots_status_list))),
        (100, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()
