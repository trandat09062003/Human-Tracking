import cv2
import os
import time
import logging
from ultralytics import YOLO
from roboflow import Roboflow

# --- CONFIGURATION ---
MODEL_PATH = 'best.pt'
VIDEO_SOURCE = '2.webm'
EXTRACT_DIR = './auto_labels'
FRAME_STRIDE = 30  # Extract 1 frame every 30 frames (~1 sec) to avoid heavy redundancy
CONF_THRESHOLD = 0.5

# Roboflow config
API_KEY = "le73LYaS0c1EiJwTNssZ"
WORKSPACE = "facedetection-uqkmv"
PROJECT = "human_26"

# Create storage directories
os.makedirs(f"{EXTRACT_DIR}/images", exist_ok=True)
os.makedirs(f"{EXTRACT_DIR}/labels", exist_ok=True)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def auto_label():
    logging.info(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logging.error("Cannot open video!")
        return

    frame_count = 0
    saved_count = 0
    
    logging.info("Starting automated extraction and labeling...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        if frame_count % FRAME_STRIDE == 0:
            # Run detection
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            
            if len(results.boxes) > 0:
                timestamp = int(time.time() * 1000)
                base_name = f"auto_{timestamp}_{frame_count}"
                img_path = f"{EXTRACT_DIR}/images/{base_name}.jpg"
                txt_path = f"{EXTRACT_DIR}/labels/{base_name}.txt"
                
                # Save image
                cv2.imwrite(img_path, frame)
                
                # Save labels (YOLO format)
                with open(txt_path, 'w') as f:
                    for box in results.boxes:
                        # YOLO: class x_center y_center width height (normalized)
                        coords = box.xywhn[0].tolist()
                        class_id = int(box.cls[0])
                        f.write(f"{class_id} {' '.join(map(str, coords))}\n")
                
                saved_count += 1
                if saved_count % 10 == 0:
                    logging.info(f"Extracted {saved_count} images automatically...")

        frame_count += 1
        
    cap.release()
    logging.info(f"Extraction complete. Total: {saved_count} images.")
    return saved_count

def upload_to_roboflow():
    logging.info("Connecting to Roboflow to upload new data...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    
    image_dir = f"{EXTRACT_DIR}/images"
    label_dir = f"{EXTRACT_DIR}/labels"
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    uploaded = 0
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
        
        if os.path.exists(label_path):
            try:
                project.upload(
                    image_path=img_path,
                    annotation_path=label_path,
                    annotation_labelmap={"0": "person"},
                    is_prediction=True, # Mark as automated labeling
                    overwrite=True
                )
                uploaded += 1
                if uploaded % 10 == 0:
                    logging.info(f"Successfully uploaded {uploaded}/{len(image_files)} images.")
            except Exception as e:
                logging.error(f"Error uploading {img_file}: {e}")
                
    logging.info(f"🎉 Upload complete! Added {uploaded} images to Roboflow.")

if __name__ == "__main__":
    count = auto_label()
    if count > 0:
        upload_to_roboflow()
    else:
        logging.warning("No humans found in video for extraction.")
