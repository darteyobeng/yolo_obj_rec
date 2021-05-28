import cv2
import numpy as np 


net = cv2.dnn.readNet('yolov3.weights','yolov3-416.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()
    

img = cv2.imread('image2.jpg')
height, width, _ = img.shape


blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)


boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidnce = scores[class_id]
        if confidence > 0.5:
            center_X = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            x = int(center_X - w/2)
            y = int(center_y - h/2)
            
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)
            
            


print(len(boxes))

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
   