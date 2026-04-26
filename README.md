# Person ReID & Multi-Camera Tracking System

## dev/creator = tubakhxn

---

## What is this project?
This project is a **Person Re-Identification (ReID) and Multi-Camera Tracking System**. It tracks people across multiple video feeds, assigning consistent global IDs to individuals as they move between different camera views. The system uses deep learning for person detection (YOLOv8), feature extraction (ResNet18), and cross-camera identity matching, enabling robust tracking in surveillance, retail analytics, and smart environments.

### Key Features
- Detects and tracks people in multiple video files simultaneously
- Assigns consistent global IDs to individuals across cameras
- Generates heatmaps and trajectory visualizations
- Entry/exit counting and dwell time estimation
- Outputs annotated video files for each camera

---

## How to fork this project
1. **Download or clone the repository** to your local machine.
2. Place your camera video files (e.g., `cam1.mp4`, `cam2.mp4`) in the project folder.
3. Run the script using Python:
   ```bash
   python "Reid multicam.py" --cams cam1.mp4 cam2.mp4
   ```
4. Optionally, adjust parameters such as processing width or ReID threshold:
   ```bash
   python "Reid multicam.py" --cams cam1.mp4 cam2.mp4 --width 640 --threshold 0.65
   ```

---

## Relevant Wikipedia Links
- [Person re-identification](https://en.wikipedia.org/wiki/Person_re-identification)
- [Object tracking](https://en.wikipedia.org/wiki/Object_tracking)
- [YOLO (object detection)](https://en.wikipedia.org/wiki/You_Only_Look_Once)
- [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network)
- [Multi-camera tracking](https://en.wikipedia.org/wiki/Multiple_camera_tracking)

---

For any questions or contributions, please contact the creator.
