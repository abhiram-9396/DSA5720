import cv2
import numpy as np
import time

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Use OpenCV backend and CUDA if available
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Change to DNN_TARGET_CUDA if using GPU

# Class names
classes = ["One", "Two", "Three", "Four", "Five"]

# Get output layer names
layer_names = net.getLayerNames()
unconnected = net.getUnconnectedOutLayers()
output_layers = [layer_names[i[0] - 1] if isinstance(i, (list, tuple, np.ndarray)) else layer_names[i - 1] for i in unconnected]

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Use smaller input size for performance
input_size = (320, 320)

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to grab frame.")
        continue

    height, width = frame.shape[:2]

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:  # Lowered threshold slightly
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    # Draw detections
    for i in range(len(indexes)):
        idx = indexes[i][0] if isinstance(indexes[i], (list, tuple, np.ndarray)) else indexes[i]
        x, y, w, h = boxes[idx]
        label = str(classes[class_ids[idx]])
        conf = str(round(confidences[idx] * 100, 2)) + "%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # FPS calculation
    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Optimized Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()