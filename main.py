# Impoting libraries
import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Creating the classRealTimeTracker


class RealTimeTracker:
    def __init__(
            self, model_path: str = "model\\yolov8s.pt",  # fast but also light weight
            confidance: float = 0.5
    ):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=100
        )  # Initializing the DeepSort tracker and YOLO model
        self.model = YOLO(model_path)
        self.cap = cv.VideoCapture(0)  # Taking input via webcam
        self.selected_id = None

# An on event function for the left mouse click
    def select_object(self, event, x, y, param):

        if event == cv.EVENT_LBUTTONDOWN:
            for track in param:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_id = track.track_id
                    print(f"Selected object with ID: {self.selected_id}")
                    break

# The main run function
    def run(self):
        while True:
            ret, frame = self.cap.read()
            # Excluded Person from the prediction list of YOLO
            # as the task specified only objects
            # also to prevent processing issues due to the poor quality of the webcam
            detection_results = self.model.predict(
                source=frame, conf=0.6, iou=0.5,
                classes=[i for i in range(len(self.model.names)) if self.model.names[i] != "person"], verbose=False
            )[0]

            detections = []

            for box in detection_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label = self.model.names[cls]

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

            self.tracks = self.tracker.update_tracks(detections, frame=frame)

            # Draw the bounding boxes and writing object names and IDs to be distinguished
            for track in self.tracks:
                if not track.is_confirmed():
                    continue
                if self.selected_id is not None and track.track_id != self.selected_id:
                    continue  # Skip all other IDs
                track_id = track.track_id

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cv.putText(
                    frame, f"{label} {track_id}", (x1, y1-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            # Showing the final frame window
            cv.imshow("Frame", frame)
            cv.setMouseCallback("Frame", self.select_object, self.tracks)
            key = cv.waitKey(30)
            if key == 27:
                break

        self.cap.release()
        cv.destroyAllWindows()


# Making an object from the class built and running the program
real_time_tracker = RealTimeTracker()
real_time_tracker.run()
