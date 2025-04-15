import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = ["One", "Two", "Three", "Four", "Five"]

layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

unconnected = net.getUnconnectedOutLayers()
output_layers = [layer_names[i[0] - 1] if isinstance(i, (list, tuple, np.ndarray)) else layer_names[i - 1] for i in unconnected]


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # for i in indexes.flatten():
    #     x, y, w, h = boxes[i]
    #     label = str(classes[class_ids[i]])
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(frame, label, (x, y - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Safely iterate over indexes regardless of format
    for i in range(len(indexes)):
        idx = indexes[i][0] if isinstance(indexes[i], (list, tuple, np.ndarray)) else indexes[i]
        x, y, w, h = boxes[idx]
        label = str(classes[class_ids[idx]])
        conf = str(round(confidences[idx] * 100, 2)) + "%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow("YOLOv3 Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()