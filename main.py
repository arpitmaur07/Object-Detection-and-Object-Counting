import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
# open dataset file
with open("coco.names", "r") as f:
    classes = f.read().splitlines()
# Loading image

cap = cv2.VideoCapture(0)
# resize = cv2.resize(cap, (640, 800))
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()

    height, width, _ = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    # Showing informations on the screen
    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Object detected
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    def countobject(whole_track_list, index_of_track):
        count = 0
        for p in whole_track_list:

            if (p == index_of_track):
                count += 1
        return count

    def numOfsameIndex(whole_redundant_list, item_id):
        count1 = 0
        for q in whole_redundant_list:
            if (q == item_id):
                count1 += 1
        return count1

    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    track = []
    redundant = []
    countlists = []

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        # print(label)
        # print(class_ids[i])
        track.append(class_ids[i])

        confidence = str(round(confidences[i], 2))
        color = colors[i]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 90), 1)
        # print(track)

        redundant = []
        countlists = []
        size = len(track)

        # print(size)
        # print(redundantsize)

        i = 0
        while i < size:

            redundantsize = len(redundant)
            # print("lenth of redundant array",redundantsize)
            if redundantsize == 0:
                totalobject = countobject(track, track[i])
                # print("object index and total number of occurace is",track[i], totalobject)
                redundant.append(track[i])
                countlists.append(totalobject)
                i += 1

            else:
                rslt = numOfsameIndex(redundant, track[i])

                if (rslt == 0):
                    totalobject = countobject(track, track[i])
                    # print("object id and total number of occurace is",track[i], totalobject)
                    # text="obj1".format(totalobject)
                    # cv2.putText(img,text,(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
                    redundant.append(track[i])
                    countlists.append(totalobject)
                    i += 1
                else:
                    i += 1
                    
    overlay = img.copy()
    # Rectangle parameters
    x, y, w, h = 380, 2, 300, 500
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)
    alpha = 0.6 # Transparency factor.
    image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)      
    
    # cv2.putText(image_new, 'SMART CONSTRUCTION SITES', (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,252,124), 2)
    # cv2.putText(image_new, 'Site: A02', (410, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    # cv2.putText(image_new, 'Site ID:', (410, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    # cv2.putText(image_new, 'Modes: PPE compliance operation', (410, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    loopsize = len(countlists) 
    x = 410
    y = 200

    for i in range(loopsize):
        label = str(classes[redundant[i]]).capitalize() 
        count = str((countlists[i]))
        cv2.putText(image_new , label, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 1)
        x += 150
        cv2.putText(image_new , count, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        x = x - 150
        y += 40
        
    cv2.imshow('Camera', image_new)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()