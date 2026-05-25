import cv2
import time
import logging
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_SOURCE = r'C:\Users\DELL\OneDrive - Hanoi University of Science and Technology\Desktop\Human_Tracking\Data\Screen Recording 2026-05-26 000928.mp4'  # Path to your input video
MODEL_PATH = 'best.pt'

# Professional logging configuration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HumanTracker:
    def __init__(self, model_path=MODEL_PATH, tracker_config='custom_tracker.yaml'):
        """
        Initialize the tracker with YOLO model and tracker configuration.
        """
        logging.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        logging.info(f"Using tracker configuration: {tracker_config if tracker_config else 'Default YOLO'}")

    def process_video(self, source, output_path=None):
        """
        Process video: Track humans and save video output.
        """
        if output_path is None:
            filename = os.path.splitext(os.path.basename(source))[0]
            output_path = f"{filename}_tracked.mp4"
            
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error(f"Cannot open video source: {source}")
            return

        # Video metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Processing: {width}x{height} @ {fps_video} FPS")

        # Setup Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

        prev_time = 0
        frame_count = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1

                # 1. Human Tracking with custom/default tracker
                if self.tracker_config:
                    results = self.model.track(
                        frame, 
                        persist=True, 
                        classes=[0], 
                        tracker=self.tracker_config, 
                        verbose=False
                    )[0]
                else:
                    results = self.model.track(
                        frame, 
                        persist=True, 
                        classes=[0], 
                        verbose=False
                    )[0]
                
                # 2. Annotation & Counting
                if results.boxes.id is not None:
                    track_ids = results.boxes.id.int().cpu().tolist()
                    annotated_frame = results.plot()
                    count = len(track_ids)
                    cv2.putText(annotated_frame, f"People: {count}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    annotated_frame = frame

                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Show & Save
                try:
                    cv2.imshow("Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass
                
                out.write(annotated_frame)

        finally:
            cap.release()
            out.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass
            logging.info(f"Processing complete. Video saved as: {output_path}")

if __name__ == "__main__":
    # CHỌN GIẢI PHÁP TRACKER TẠI ĐÂY:
    # 1. 'custom_tracker.yaml'  -> ByteTrack (Tối ưu nhất, khuyên dùng để chống nhảy ID)
    # 2. 'custom_botsort.yaml'  -> BoT-SORT (Tối ưu tăng bộ nhớ theo vết lên gấp 4 lần)
    # 3. None                  -> Tracker mặc định của Ultralytics YOLO
    TRACKER_CONFIG = 'custom_tracker.yaml'

    tracker = HumanTracker(tracker_config=TRACKER_CONFIG)
    tracker.process_video(source=VIDEO_SOURCE)

