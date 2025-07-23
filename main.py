# Importing libraries
import cv2 as cv
from ultralytics import YOLO
import time
import threading
from queue import Queue, Empty


# Creating the classRealTimeTracker
class RealTimeTracker:
    def __init__(
            self, model_path: str = "model\\yolov8s.pt",  # fast but also light weight
            confidence: float = 0.5,
            target_fps: int = 30,
            process_every_n_frames: int = 1
    ):
        self.confidence = confidence
        self.target_fps = target_fps
        self.process_every_n_frames = process_every_n_frames
        self.model = YOLO(model_path)  # Initializing the YOLO model

        # Pre-calculate excluded classes for better performance
        self.valid_classes = [i for i in range(
            len(self.model.names)) if self.model.names[i] != "person"]

        self.cap = cv.VideoCapture(0)  # Taking input via webcam
        self.selected_id = None

        # Optimize camera settings for better performance
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv.CAP_PROP_FPS, target_fps)

        # Try to set camera format to improve performance
        try:
            self.cap.set(cv.CAP_PROP_FOURCC,
                         cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except:
            pass  # Some cameras don't support MJPG

        # Threading components for responsiveness
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=1)
        self.running = False
        self.detection_thread = None

        # Performance tracking
        self.last_detection_result = None

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

    def detection_worker(self):
        """Runs YOLO detection in a separate thread to prevent UI blocking"""
        frame_skip_counter = 0

        while self.running:
            try:
                # Get frame from queue (non-blocking)
                frame = self.frame_queue.get(timeout=0.1)

                # Apply frame skipping logic in the detection thread
                frame_skip_counter += 1
                if frame_skip_counter % self.process_every_n_frames != 0:
                    continue

                # Run YOLO detection
                detection_results = self.model.track(
                    source=frame,
                    conf=self.confidence,
                    iou=0.5,
                    tracker="Model\\botsort.yaml",
                    classes=self.valid_classes,
                    verbose=False,
                    persist=True,
                    half=True,
                    imgsz=416,
                    device='0' if self.model.device.type == 'cuda' else 'cpu'
                )[0]

                # Put result in queue (replace old result if queue is full)
                try:
                    self.result_queue.put_nowait(detection_results)
                except:
                    try:
                        self.result_queue.get_nowait()  # Remove old result
                        self.result_queue.put_nowait(detection_results)
                    except Empty:
                        pass

            except Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(0.1)

# The main run function
    def run(self):
        # Test detection before starting threaded mode
        ret, test_frame = self.cap.read()
        if ret:
            try:
                test_results = self.model.track(
                    source=test_frame,
                    conf=self.confidence,
                    classes=self.valid_classes,
                    verbose=False
                )[0]
                if test_results.boxes is None:
                    print("No objects detected in test frame")
            except Exception as e:
                print(f"Detection initialization failed: {e}")
                return

        self.running = True

        # Start detection thread
        self.detection_thread = threading.Thread(
            target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        time.sleep(0.5)

        # Initialize performance tracking
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        frame_time_target = 1.0 / self.target_fps

        print("Real-time object tracker initialized successfully!")
        print("- Click on any detected object to track it exclusively")
        print("- Click elsewhere to deselect and track all objects")
        print("- Press ESC to exit")

        try:
            while self.running:
                loop_start_time = time.time()
                ret, frame = self.cap.read()

                if not ret:
                    break

                frame_count += 1
                fps_counter += 1

                # Calculate FPS every second
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    current_fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time

                # Send frame to detection thread
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except:
                    pass  # Queue full, skip this frame

                # Get latest detection results
                detection_results = None
                try:
                    detection_results = self.result_queue.get_nowait()
                    self.last_detection_result = detection_results
                except Empty:
                    detection_results = self.last_detection_result

                tracked_objects = 0

                if detection_results is not None and detection_results.boxes is not None:
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

                        # Draw bounding box and center point
                        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv.circle(frame, (cx, cy), 5, color, -1)

                        label = f"{class_name} ID:{track_id} ({conf:.2f})"
                        cv.putText(frame, label, (x1, y1 - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display info text
                info_parts = [
                    f"Frame: {frame_count}", f"Objects: {tracked_objects}", f"FPS: {current_fps:.1f}"]
                if self.selected_id is not None:
                    info_parts.append(f"Selected ID: {self.selected_id}")
                info_text = " | ".join(info_parts)

                cv.putText(frame, info_text, (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv.putText(frame, "Click object to track | ESC to exit", (10,
                           frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Display frame
                cv.imshow("Real-Time Object Tracker", frame)
                cv.setMouseCallback("Real-Time Object Tracker", self.select_object,
                                    detection_results.boxes if detection_results is not None and detection_results.boxes is not None else [])

                key = cv.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

                # Control frame rate
                loop_time = time.time() - loop_start_time
                if loop_time < frame_time_target:
                    time.sleep(min(0.01, frame_time_target - loop_time))

        finally:
            self.running = False
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)
            self.cap.release()
            cv.destroyAllWindows()
            print("Tracker stopped!")


if __name__ == "__main__":
    real_time_tracker = RealTimeTracker(
        confidence=0.5,  # Lower confidence to detect more objects
        target_fps=30,   # Good performance target
        process_every_n_frames=1  # Process every frame for best detection
    )
    real_time_tracker.run()
