# Install dependencies first
# pip install ultralytics opencv-python shapely numpy

from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Point, Polygon

# 1️⃣ Load YOLO pretrained model
model = YOLO("yolov8s.pt")  # Small model, fast

# 2️⃣ Load image
image_path = "1212.png"  # Replace with your image path
frame = cv2.imread(image_path)

# Check if image loaded
if frame is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# 3️⃣ Define queue ROI (polygon points)
queue_roi = [(100, 200), (500, 200), (500, 600), (100, 600)]  # Replace with your ROI
roi_poly = Polygon(queue_roi)

def point_in_roi(x, y):
    """Check if a point is inside the queue ROI polygon"""
    return roi_poly.contains(Point(x, y))

# 4️⃣ Run YOLO detection
results = model(frame)

count_in_queue = 0

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        # COCO class 0 = person
        if cls_id == 0 and conf > 0.5:
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            if point_in_roi(x_center, y_center):
                count_in_queue += 1
                # Green box = inside queue
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
                # Red box = outside queue
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# 5️⃣ Draw ROI polygon
pts = np.array(queue_roi, dtype=np.int32)
cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

# 6️⃣ Show count on image
cv2.putText(frame, f"Queue Count: {count_in_queue}", (50,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

print(f"Number of people in queue: {count_in_queue}")

# 7️⃣ Display the annotated image
cv2.imshow("Queue Counting", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: save annotated image
cv2.imwrite("queue_annotated.jpg", frame)
