# Impoting libraries
import cv2 as cv
from ultralytics import YOLO
import time


# Creating the classRealTimeTracker
class RealTimeTracker:
    def __init__(
            self, model_path: str = "model\\yolov8s.pt",  # fast but also light weight
            confidance: float = 0.5
    ):
        self.confidance = confidance
        self.model = YOLO(model_path)  # Initializing the YOLO model
        self.cap = cv.VideoCapture(0)  # Taking input via webcam
        self.selected_id = None

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv.CAP_PROP_FPS, 30)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

# An on event function for the left mouse click
    def select_object(self, event, x, y, flags, param):

        if event == cv.EVENT_LBUTTONDOWN:
            boxes = param
            if boxes is None:
                return
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if box.id is not None:
                        track_id = int(box.id[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        self.selected_id = track_id
                        print(
                            f"Selected {class_name} with ID: {self.selected_id}")
                        return
            if self.selected_id is not None:
                self.selected_id = None
                print("Deselected object - tracking all objects")

# The main run function
    def run(self):

        # Initializing variables used in the displayed window
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0

        print("Real-time object tracker initialized successfully!")
        print("- Click on any detected object to track it exclusively")
        print("- Click elsewhere to deselect and track all objects")
        print("- Press ESC to exit")

        while True:
            ret, frame = self.cap.read()

            frame_count += 1
            fps_counter += 1

            # Excluded Person from the prediction list of YOLO
            # as the task specified only objects
            # also to prevent processing issues due to the poor quality of the webcam
            detection_results = self.model.track(
                source=frame, conf=self.confidance, iou=0.5,
                tracker="Model\\botsort.yaml",
                classes=[i for i in range(
                    len(self.model.names)) if self.model.names[i] != "person"],
                verbose=False,
                persist=True
            )[0]

            tracked_objects = 0

            if detection_results.boxes is not None:
                for box in detection_results.boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id[0])

                    if self.selected_id is not None and track_id != self.selected_id:
                        continue

                    tracked_objects += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    color = (0, 255, 0) if self.selected_id == track_id else (
                        0, 0, 255)
                    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv.circle(frame, (cx, cy), 5, color, -1)

                    label = f"{class_name} ID:{track_id} ({conf:.2f})"
                    cv.putText(frame, label, (x1, y1 - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            info_text = f"Frame: {frame_count} | Objects: {tracked_objects} | FPS: {current_fps:.1f}"
            if self.selected_id is not None:
                info_text += f" | Selected ID: {self.selected_id}"

            cv.putText(frame, info_text, (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, "Click object to track | ESC to exit", (10,
                       frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Showing the final frame window
            cv.imshow("Real-Time Object Tracker", frame)
            cv.setMouseCallback("Real-Time Object Tracker", self.select_object,
                                detection_results.boxes if detection_results.boxes is not None else [])
            key = cv.waitKey(30)
            if key == 27:
                break

        self.cap.release()
        cv.destroyAllWindows()
        print("Tracker stopped!")


# Making an object from the class built and running the program
real_time_tracker = RealTimeTracker()
real_time_tracker.run()
