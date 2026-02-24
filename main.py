###################################################################################################
# This script performs object detection and tracking on a video using YOLOv8 and a DeepSORT Tracker.
# It saves the output video with bounding boxes, a speed-based trajectory video with a color bar
# indicating speed, and a trajectory video showing the paths of tracked objects. It also computes 
# time to process the frame. The script also saves tracking results in a text file in MOT format.
###################################################################################################

import os
import random
import numpy as np
import cv2
from ultralytics import YOLO
from matplotlib import cm
import matplotlib.pyplot as plt
from tracker import Tracker
import time

video_path = os.path.join('.', 'data', 'smooth_best_21.mp4')
video_out_path = os.path.join('.', 'out.mp4')
speed_video_out_path = os.path.join('.', 'speed_output.mp4')
trajectory_video_out_path = os.path.join('.', 'trajectory_output.mp4')

# Create directories for saving detections and tracking results
os.makedirs('tracking_results', exist_ok=True)
track_output_path = 'tracking_results/sm21_def.txt'

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))
speed_out = cv2.VideoWriter(speed_video_out_path, fourcc, fps, (frame_width + 60 + 60, frame_height))  # Adjust width for color bar
trajectory_out = cv2.VideoWriter(trajectory_video_out_path, fourcc, fps, (frame_width, frame_height))

# Load the YOLOv5 model
model = YOLO("model_obj_detector/best.pt", "v8")                                #turkeyDetect_bs8_cst.pt", "v8")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5

# Initialize a blank image for the trajectories
trajectory_image = 255 * np.ones((frame_height, frame_width, 3), dtype=np.uint8)
speed_frame = 255 * np.ones((frame_height, frame_width, 3), dtype=np.uint8)

# To store the previous center points for each track
prev_centers = {}

# Create a colormap
cmap = cm.get_cmap('Reds')
norm = plt.Normalize(0, 30) 

with open(track_output_path, 'w') as track_file:
    
    print("Start tracking:")

    while cap.isOpened:

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        t1 = time.time()

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

            tracker.update(frame, detections)

            for track in tracker.tracks:
                
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                score = track.score

                # Save each tracked object to the tracks file in MOT format
                track_file.write(f"{frame_id},{track_id},{x1},{y1},{x2 - x1},{y2 - y1},{score},1,1.0\n")

                # Calculate the current center of the bounding box
                current_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

                # Draw the track ID on the current frame near the object
                cv2.putText(frame, str(track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[track_id % len(colors)], 2)

                # Draw the trajectory line on the trajectory image
                if track_id in prev_centers:
                    prev_center = prev_centers[track_id]
                    distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
                    speed = distance * 1 #fps

                    # Get a color from the colormap based on speed
                    color = tuple([int(c * 255) for c in cmap(norm(speed))[:3]])

                    cv2.line(speed_frame, prev_center, current_center, color, 2)  # colors[track_id % len(colors)]
                    cv2.line(trajectory_image, prev_center, current_center, colors[track_id % len(colors)], 2)  

                # Update the previous center
                prev_centers[track_id] = current_center

        t2 = time.time()
        print("Total time taken per frame:", (t2-t1))
        # Add a vertical color bar to the right of the speed_frame
        color_bar_height = frame_height
        color_bar_width = 60
        color_bar = np.zeros((color_bar_height, color_bar_width, 3), dtype=np.uint8)

        for i in range(color_bar_height):
            color = cmap(i / color_bar_height)[:3]
            color_bar[i, :] = [int(c * 255) for c in color]

        # Create a blank area for the scale labels next to the color bar
        label_width = 60
        labeled_color_bar = np.hstack((color_bar, np.ones((color_bar_height, label_width, 3), dtype=np.uint8) * 255))

        # Add scale labels (0, 10, 20, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (0, 0, 0)  # Black text

        # Define label positions
        label_positions = [30, 25, 20, 15, 10, 5, 0]
        for i, label in enumerate(label_positions):
            y_position = int(color_bar_height - (i + 1) * (color_bar_height / (len(label_positions) + 1))) + 10
            cv2.putText(labeled_color_bar, str(label), (color_bar_width + 10, y_position), font, font_scale, text_color, thickness, cv2.LINE_AA)


        # Combine speed_frame and labeled_color_bar
        speed_frame_with_bar = np.hstack((speed_frame, labeled_color_bar))

        #cv2.imshow("frame",frame)  
        ## Show the trajectory image in a separate window
        #cv2.imshow("trajectories", trajectory_image)
        #cv2.imshow("Speed-Based Trajectories", speed_frame_with_bar)

        out.write(frame)
        speed_out.write(speed_frame_with_bar)  # Speed-based trajectory video with color bar
        trajectory_out.write(trajectory_image)  # Trajectory video with default colors


        key = cv2.waitKey(1)
        if  key == 27:
            break

cap.release()
out.release()
speed_out.release()
trajectory_out.release()
cv2.destroyAllWindows()