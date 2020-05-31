import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
with open("yolov3.names","r") as f:
     classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


img = cv2.imread("cars_1.jpg")
img = cv2.resize(img,None,fx=0.6,fy=0.6)
height,width,channels = img.shape


blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)



net.setInput(blob)
outs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[6:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.4)
print(indexes)
v = len(indexes)
print(v)
for i in range(len(boxes)):
    if i in indexes:
       x,y,w,h = boxes[i]
       label = classes[class_ids[i]]
       print(label)
       cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
