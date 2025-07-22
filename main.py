import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

model = YOLO("model\\yolov8s.pt")

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.3,
    nn_budget=100
)

cap = cv.VideoCapture(0)

selected_id = None


def select_object(event, x, y, flags, param):
    global selected_id
    if event == cv.EVENT_LBUTTONDOWN:

        for track in param:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = track.track_id
                print(f"Selected object with ID: {selected_id}")
                break


while True:
    ret, frame = cap.read()

    detection_results = model.predict(
        source=frame, conf=0.6, iou=0.5,
        classes=[i for i in range(len(model.names)) if model.names[i] != "person"], verbose=False
    )[0]

    detections = []

    for box in detection_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = model.names[cls]

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        if selected_id is not None and track.track_id != selected_id:
            continue  # Skip all other IDs
        track_id = track.track_id

        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv.putText(
            frame, f"{label} {track_id}", (x1, y1-10),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    cv.imshow("Frame", frame)
    cv.setMouseCallback("Frame", select_object, tracks)
    key = cv.waitKey(30)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
