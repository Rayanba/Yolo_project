import cv2
import cvzone
from ultralytics import YOLO
import math
import numpy as np

from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
model = YOLO('models/best.pt')
classNames = ['0', '1', '2', '3', 'accident', 'bike', 'bus', 'car', 'cars', 'carwithplate', 'person', 'plate', 'car']
model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

# Initialize lane detection model

lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    # success, img = cap.read()
    results = model(frame, stream=True)
    # Detect the lanes

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            conf = math.ceil((box.conf[0]*100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0,x1), max(35,y1)), scale = .7, thickness = 1)
    output_img = lane_detector.detect_lanes(frame)
    #
    # cv2.imshow("Detected lanes", img)
    # cv2.waitKey(1)

    cv2.imshow("Detected lanes", output_img)
    cv2.waitKey(1)


