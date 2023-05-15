import cv2
import numpy as np
import os
import time

# Load pre-trained model
model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Set preferable target and backend
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Set input scale and input swap RB
input_scale = 1.0/255
input_swap_rb = True

# Load class labels
with open("coco.names", "rt") as f:
    labels = f.read().rstrip("\n").split("\n")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create directory for saving detected faces
if not os.path.exists("DetectedFaces"):
    os.makedirs("DetectedFaces")

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, input_scale, (416, 416), swapRB=input_swap_rb, crop=False)
    model.setInput(blob)
    layer_outputs = model.forward(model.getUnconnectedOutLayersNames())
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * frame.shape[1]), int(detection[1] * frame.shape[0])
                width, height = int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])
                left, top = int(center_x - width / 2), int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Draw bounding boxes and labels
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            label = labels[class_ids[i]]
            confidence = confidences[i]
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if label == 'person' and confidence >= 0.9:
                person_frame = frame[top:top+height, left:left+width]
                # Save the detected person image
                filename = f"DetectedFaces/person_{time.time()}.png"
                if person_frame.size > 0:
                    cv2.imwrite(filename, person_frame)

    # Show frame
    cv2.imshow("Object detection", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
