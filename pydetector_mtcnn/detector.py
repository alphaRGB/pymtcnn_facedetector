import numpy as np
import cv2
import pydetector


def detect_image(img_name):
    image = cv2.imread(img_name)
    assert image
    # Create a FaceDetector 
    face_detector =  pydetector.FaceDetector(model_path='./models', num_thread=1, scale=0.25)
    # Detect 
    boxes = face_detector.detect(img_bgr=image)
    if len(boxes > 0):
        for item in boxes:
            cv2.rectangle(image, (item.x, item.y), (item.x + item.width, item.y + item.height), (0, 255, 255), 2)
    cv2.imshow('face-detect', image)
    cv2.waitKeyEx(0)



def detect_video(video_file):
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened()
    
    face_detector =  pydetector.FaceDetector(model_path='./models', num_thread=1, scale=0.25)
    meter = cv2.TickMeter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        meter.reset()
        meter.start()
        boxes = face_detector.detect(img_bgr=frame)
        meter.stop()
        for item in boxes:
            cv2.rectangle(frame, (item.x, item.y), (item.x + item.width, item.y + item.height), (0, 255, 255), 2)
        cv2.imshow('face-detect', frame)
        print('time={:.3}ms'.format(meter.getTimeMilli()))
        cv2.waitKeyEx(33)


if __name__ == '__main__':
    detect_video(video_file=0)