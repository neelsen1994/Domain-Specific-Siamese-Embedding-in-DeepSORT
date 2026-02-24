###############################################################################################
# This script creates a dataset for Siamese network training by extracting and saving
# cropped images of detected objects from a video based on provided bounding box annotations.
# The annotations should be in MOT format or a CSV file with frame-wise bounding box information.
# The cropped images are saved in a structured directory format for easy access during training.
###############################################################################################

import cv2
import os
import pandas as pd
from tqdm import tqdm

# Paths
video_path = "./data/smooth_best_21.mp4"
annotations_path = "/home/debayansen/projects/AIServicePlatform/services/ai-inference/object-tracker/dnn_tracker_deepsort/TrackEval/data/gt/mot_challenge/Turkey-test/seq3/gt/gt.txt"  # MOT format or CSV file with bbox info
output_dir = "dataset_siam_21"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load the annotation file
def load_annotations(annotations_path):
    """
    Load bounding box annotations.
    Assumes format: frame_id, object_id, x, y, width, height, confidence
    """
    cols = ["frame_id", "object_id", "x", "y", "w", "h", "confidence", "class", "visibility"]
    df = pd.read_csv(annotations_path, header=None, names=cols)
    return df

# Crop bounding boxes and save images
def crop_and_save(video_path, annotations, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    pbar = tqdm(total=len(annotations["frame_id"].unique()) / 100)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        if frame_id % 100 != 0:
            continue
        current_annotations = annotations[annotations["frame_id"] == frame_id]
        
        for _, row in current_annotations.iterrows():
            object_id = int(row["object_id"])
            x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
            
            # Crop the bounding box
            cropped_img = frame[y:y+h, x:x+w]
            if cropped_img.size == 0:
                continue  # Skip empty crops
            
            # Create directory for the object_id
            obj_dir = os.path.join(output_dir, f"object_{object_id}")
            os.makedirs(obj_dir, exist_ok=True)
            
            # Save cropped image
            img_path = os.path.join(obj_dir, f"frame_{frame_id}.jpg")
            cv2.imwrite(img_path, cropped_img)
        pbar.update(1)
        
    pbar.close()
    cap.release()

# Main
annotations = load_annotations(annotations_path)
crop_and_save(video_path, annotations, output_dir)
print("Dataset creation complete.")
