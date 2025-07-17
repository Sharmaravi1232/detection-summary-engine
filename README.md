Detection Summary Engine
Detection Summary Engine is a streamlined video analysis tool that uses pretrained object detection models such as YOLOv5 or Faster R-CNN to process short .mp4 videos. It extracts and summarizes detection data at regular frame intervals, offering structured outputs and visual insights.

Key Features
Uses pretrained models (YOLOv5, Faster R-CNN)

Processes every 5th frame of a video (15–20 seconds duration)

Outputs per-frame JSON data including:

Detected class labels

Bounding box coordinates

Confidence scores

Computes total object counts per class

Identifies the frame with the highest class diversity

Generates a bar chart of object frequency

(Optional) Saves annotated frames or compiles a video with visual overlays

detection-summary-engine/
├── models/            # Model loading and config
├── utils/             # Helper functions
├── outputs/           # Annotated frames, charts, compiled video
├── video_input/       # Input video files
├── main.py            # Main pipeline script
├── requirements.txt   # Python dependencies
└── README.md

pip install -r requirements.txt
Requirements
torch, torchvision

opencv-python

matplotlib

numpy, pandas
python main.py --video ./video_input/sample.mp4 --model yolo

ptional arguments:

--save-frames — Save annotated frames

--save-video — Save output video with detections

--frame-interval — Frame skip interval (default: 5)

Output
JSON metadata per frame

Object frequency visualization (object_frequency.png)

Frame index with maximum class diversity

Optional: annotated images or compiled output video

Supported Models
YOLOv5 (via Ultralytics PyTorch Hub)

Faster R-CNN (via torchvision.models.detection)

Custom model integration is supported with minimal modifications.
Future Enhancements
Live webcam feed support

Lightweight model integration (e.g., YOLOv8-nano, MobileNet SSD)

Exportable summary reports (PDF/HTML)

Optimized real-time inference




