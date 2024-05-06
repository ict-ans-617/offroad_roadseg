import cv2
import time


def open_video_capture(device=0):
    video_capture = cv2.VideoCapture(device)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not video_capture.isOpened():
        print("Cannot open camera")
        exit()
    print(f"{video_capture.get(cv2.CAP_PROP_BUFFERSIZE) = }")
    return video_capture


def read_video_frame(video_capture, ):
    ret, frame = video_capture.read()
    capture_time = time.time()
    print(f"{capture_time = }")
    return ret, frame
