# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
import os
import time
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# BODY_PARTS = {  "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "Background": 18 }

# POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"] ]



inWidth = args.width
inHeight = args.height

# model_restored = load_model('D:/ChinhResources/snake/model/black_skeleton/black_100.h5')
# label_class = {0: 'down', 1: 'left', 2: 'normal', 3: 'right', 4: 'up'}

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture(args.input if args.input else 0)

count = 1300
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    frame = cv.flip(frame,1)

    # x_crop = frameWidth/2 - inWidth/2
    # y_crop = frameHeight/2 - inHeight/2

    # frame = frame[int(y_crop):int(y_crop+inHeight), int(x_crop):int(x_crop+inWidth)]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
 
    # assert(len(BODY_PARTS) == out.shape[1])9

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3] - 130  #- x_crop
        y = (frameHeight * point[1]) / out.shape[2] #- y_crop
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    # img_black = 255 * np.zeros((inWidth, inHeight, 1), dtype = "uint8")
    img_white = np.zeros([inWidth, inHeight,3],dtype=np.uint8)
    img_white.fill(255)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        # assert(partFrom in BODY_PARTS)
        # assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(img_white, points[idFrom], points[idTo], (0, 0, 0), 3)
            # cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (255, 255, 255), cv.FILLED)
            # cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.line(frame, points[idFrom], points[idTo], (0, 0, 0), 3)

    # t, _ = net.getPerfProfile()
    # freq = cv.getTickFrequency() / 1000

    time.sleep(0.3)
    path = 'D:\ChinhResources\snake\pictures'
    cv.imwrite(os.path.join(path , f'image{count}.jpg'), img_white)
    count += 1

    # cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    # img_white = cv.resize(img_white, (96,96), interpolation = cv.INTER_AREA)
    # # img_white = cv.imread(img_white)
    # # img_white=load_img(img_white, target_size=(96, 96))

    # x=img_to_array(img_white)
    # x=np.expand_dims(x, axis=0)

    # Predict -------------------------------------------------
    # prediction = model_restored.predict(x, batch_size=10)
    # label = np.argmax(prediction, axis = 1)
    # name = label_class[label[0]]
    # proba = round(max(prediction[0])*100)
    
    # Display text about confidence rate above each box
    # text = f'{name}, {proba}%'

    # cv.putText(frame,text,(0,35),cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0),2)

    cv.imshow('OpenPose using OpenCV', frame)
