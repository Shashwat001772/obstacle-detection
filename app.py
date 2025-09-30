# app.py
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Load the YOLO model
net = cv2.dnn.readNet("yolo_model/yolov3.weights", "yolo_model/yolov3.cfg")
classes = []
with open("yolo_model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

print("YOLO model loaded successfully!")

last_announcement = None
last_announcement_time = 0
announcement_delay = 4

def generate_frames():
    global last_announcement, last_announcement_time
    
    # Using the built-in laptop webcam
    cap = cv2.VideoCapture(1) 
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam. Make sure it is not being used by another application.")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
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
        
        announcement = None
        
        if len(indexes) > 0:
            left_zone_boundary = width * 0.33
            right_zone_boundary = width * 0.66
            
            left_obstacles = 0
            center_obstacles = 0
            right_obstacles = 0
            
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                object_center_x = x + w / 2
                
                if object_center_x < left_zone_boundary:
                    left_obstacles += 1
                elif object_center_x > right_zone_boundary:
                    right_obstacles += 1
                else:
                    center_obstacles += 1
                
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            # Prescriptive Command Logic
            if center_obstacles > 0:
                if left_obstacles <= right_obstacles:
                    announcement = "Obstacle in front. Take a right."
                else:
                    announcement = "Obstacle in front. Take a left."
        
        current_time = time.time()
        if announcement:
            if announcement != last_announcement or (current_time - last_announcement_time) > announcement_delay:
                socketio.emit('announcement', {'data': announcement})
                last_announcement = announcement
                last_announcement_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, debug=True)