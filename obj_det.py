from gtts import gTTS
import IPython.display as ipd
from pydub import AudioSegment
import subprocess
import imutils
import os
import cv2
from urllib import request
import numpy as np
import time
import random
font = cv2.FONT_HERSHEY_SIMPLEX


def current_milli_time(): return int(round(time.time() * 1000))


LABELS = open("coco.names").read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

stream = request.urlopen('ip address of the esp32 cam ') #Make sure you click on the inspect element from the stream and then you'll get the exact ip address of the stream 
                                                         # it would look something like this http://192.168.1.103:81/stream
# stream = cv2.VideoCapture(0)

bts = b''
count = 0
total_fails = 10
starttime = time.time()
t1 = current_milli_time()
t2 = t1
while True:
    bts += stream.read(1024)
    a = bts.find(b'\xff\xd8')
    b = bts.find(b'\xff\xd9')
    # print(a, b)

    if a != -1 and b != -1:
        jpg = bts[a:b+2]
        bts = bts[b+2:]
        img = cv2.imdecode(np.fromstring(
            jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        # get image height, width
        (H, W) = img.shape[:2
                           ]
        # calculate the center of the image
        
        
        
        center = (W / 2, H / 2)

        # angle90 = 90
        angle180 = 180
        #angle270 = 360

        scale = 1.5

        # Perform the counter clockwise rotation holding at the center
        # 90 degrees
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated270 = cv2.warpAffine(img, M, (H, W))

        # cv2.imshow('Video', img)

        blob = cv2.dnn.blobFromImage(
            img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []
        centers = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    centers.append((centerX, centerY))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        # draw bounding boxes on the image
        colors = np.random.uniform(0, 255, size=(len(LABELS), 3))

        for i in idxs:
            i = i[0]
            x, y, w, h = boxes[i]
            classID = classIDs[i]
            color = colors[classID]
            cv2.rectangle(img, (round(x), round(y)),
                          (round(x+w), round(y+h)), color, 2)
            label = "%s: %.2f" % (LABELS[classID], confidences[i])
            cv2.putText(img, label, (x-10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Object detection", img)
            cv2.waitKey(50)

            texts = []

            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # find positions
                    centerX, centerY = centers[i][0], centers[i][1]
                    if centerX <= W/3:
                        W_pos = "left "
                    elif centerX <= (W/3 * 2):
                        W_pos = "center "
                    else:
                        W_pos = "right "

                    if centerY <= H/3:
                        H_pos = "top "
                    elif centerY <= (H/3 * 2):
                        H_pos = "mid "
                    else:
                        H_pos = "bottom "
                    texts.append(H_pos + W_pos + LABELS[classIDs[i]])
                print(texts)

                if texts:
                    description = ', '.join(texts)
                    tts = gTTS(description, lang='en')
                    tts.save('tts.mp3')
                    ipd.Audio('tts.mp3')
                    tts = AudioSegment.from_mp3("tts.mp3")

                    subprocess.call(
                        ["ffplay", "-nodisp", "-autoexit", "tts.mp3"])

        if cv2.waitKey(1) == 27:
            break
