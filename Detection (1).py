# STEP 0: Install Required Libraries
!pip install ultralytics opencv-python matplotlib seaborn -q

# STEP 1: Import Dependencies
import os
import cv2
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from ultralytics import YOLO

# STEP 2: Define Paths and Initialize Directories
video_path = '/content/sample_video.mp4'  # Input video file
frames_dir = '/content/frames'
json_output_dir = '/content/json_output'
annotated_frames_dir = '/content/annotated_frames'

# Create directories if they do not exist
for directory in [frames_dir, json_output_dir, annotated_frames_dir]:
    os.makedirs(directory, exist_ok=True)

# STEP 3: Load Pretrained YOLOv5s Model
model = YOLO('yolov5s.pt')

# STEP 4: Extract Every 5th Frame from the Video
cap = cv2.VideoCapture(video_path)
frame_interval = 5
frame_index = 0
extracted_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % frame_interval == 0:
        frame_path = os.path.join(frames_dir, f'frame_{frame_index}.jpg')
        cv2.imwrite(frame_path, frame)
        extracted_frames.append((frame_index, frame_path))

    frame_index += 1

cap.release()

# STEP 5: Perform Object Detection and Save Results
object_counts = Counter()
class_diversity = {}
detections_per_frame = {}

for frame_id, image_path in extracted_frames:
    detection = model(image_path)[0]

    current_detections = []
    unique_labels = set()

    if detection.boxes and detection.boxes.cls.numel() > 0:
        for box in detection.boxes:
            cls = int(box.cls.item())
            label = model.names[cls]
            confidence = round(box.conf.item(), 3)
            coords = [round(x, 1) for x in box.xyxy[0].tolist()]

            current_detections.append({
                "class": label,
                "bbox": coords,
                "confidence": confidence
            })

            object_counts[label] += 1
            unique_labels.add(label)

    class_diversity[frame_id] = len(unique_labels)
    detections_per_frame[f"frame_{frame_id}"] = current_detections

    # Save detection results as JSON
    with open(os.path.join(json_output_dir, f'frame_{frame_id}.json'), 'w') as json_file:
        json.dump(current_detections, json_file, indent=2)

    # Save annotated frame
    annotated_image_path = os.path.join(annotated_frames_dir, f'frame_{frame_id}_annotated.jpg')
    detection.save(filename=annotated_image_path)

# STEP 6: Save Summary of All Detections
with open('/content/detection_summary.json', 'w') as summary_file:
    json.dump(detections_per_frame, summary_file, indent=2)

# STEP 7: Plot Object Class Frequencies
if object_counts:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(object_counts.keys()), y=list(object_counts.values()), palette='viridis')
    plt.title("Detected Object Frequency")
    plt.xlabel("Object Class")
    plt.ylabel("Total Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/content/object_frequency.png')
    plt.show()
else:
    print("No objects were detected across the frames.")

# STEP 8: Identify Frame with Maximum Class Variety
if class_diversity:
    most_diverse_frame = max(class_diversity, key=class_diversity.get)
    print(f"Frame with the most class diversity: frame_{most_diverse_frame} "
          f"({class_diversity[most_diverse_frame]} unique classes)")
else:
    print("No class diversity found; possibly no detections.")

# STEP 9: Compile Annotated Frames into a Video
annotated_images = sorted(glob.glob(os.path.join(annotated_frames_dir, '*.jpg')))

if annotated_images:
    sample_img = cv2.imread(annotated_images[0])
    height, width, _ = sample_img.shape
    output_path = '/content/annotated_video.avi'
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 5, (width, height))

    for img_path in annotated_images:
        img = cv2.imread(img_path)
        video_writer.write(img)

    video_writer.release()
    print(f"Annotated video successfully saved at: {output_path}")
else:
    print("No annotated frames were found; video compilation skipped.")
