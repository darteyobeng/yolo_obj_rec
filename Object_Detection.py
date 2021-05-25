import cv2
import numpy as np 


net = cv2.dnn.readNet('yolov3.weights','yolov3-416.cfg')
classes = []
with open('name_file.names', 'r') as f:
    classes = f.read().splitlines()
    
    
    print(classes)
    
   