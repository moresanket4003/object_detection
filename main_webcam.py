import cv2
import numpy as np
import os

# Threshold and NMS settings
thres = 0.45
nms_threshold = 0.5

# Start webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 150)

# Import class names
classNames = []
classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Import config and weights files
configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))

# Set up the network
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, image = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Detect objects in the image
    classIds, confs, bbox = net.detect(image, confThreshold=thres)
    
    if len(classIds) != 0:
        # Convert to lists for NMS
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))
        print(classIds, confs, bbox)

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                
                # Access classIds without extra indexing
                cv2.putText(image, classNames[int(classIds[i]) - 1].upper(), (x + 10, y + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        print("No detections")

    # Display the output
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
