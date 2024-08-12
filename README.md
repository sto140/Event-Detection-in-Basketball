# Basketball Event Detection System

This repository contains the code and resources for a Basketball Event Detection System, developed using computer vision and machine learning techniques. The system is designed to automate the detection and tracking of key events during basketball games, such as player movements, basketball location, and successful shots. By leveraging state-of-the-art technologies like YOLOv8, OpenCV, and Python, this project aims to enhance the analysis and understanding of basketball game footage.

![image](https://github.com/user-attachments/assets/354aee58-65b5-4831-b9d6-98ba783d02a3) ![image](https://github.com/user-attachments/assets/668b7abc-8192-43a7-b267-a741d76ea320)


## Features

- **Object Detection:** Utilizes YOLOv8 to accurately detect and classify objects such as players, basketballs, and hoops in real-time.
- **Object Tracking:** Tracks the movement of detected objects throughout the game to provide continuous monitoring.
- **Shot Detection:** Identifies when a shot has been taken and determines if it was successful, enabling automatic score tracking.
- **Video Output:** Generates annotated video footage showing detected objects and events, providing a visual summary of the game.

## Installation

### Prerequisites

- Python
- pip (Python package installer)
- Git
- Roboflow
- OpenCV
- YOLO

### Project Structure
- Large Model.pt - Model train on basketball images containing key objects( basketball, players, madebaskets)
- shot_detector.py: Main script for detecting and tracking basketball events in video footage.
- train.py: Script for training the object detection model using YOLOv8.
- utils.py: provides funtionality to determine whether the basketball has been scored base on factors such as ball location.
-testmodel.py: tests the of various criteria such as precision, recall, and accuracy of detections/tracking

## Usage

### 1. Training the Model

To train the object detection model using your custom dataset:

1. **Prepare Your Dataset**: Use tools like Roboflow to annotate your dataset with the classes `basketball`, `player`, and `hoop`.
2. **Configure the Model**: Edit the `config.yaml` file to specify the dataset paths, model architecture, and other parameters.
3. **Start Training**: Run the following command to begin training:

    ```bash
    python train.py --data config.yaml --epochs 200
    ```

This will train the YOLOv8 model on your dataset for 50 epochs. The trained model weights will be saved to the `runs/train/` directory.

### 2. Testing the Model

To test the trained model on a validation or test dataset:

1. **Prepare Test Data**: Ensure your test dataset is correctly annotated and formatted as per YOLO requirements.
2. **Run the Test Script**:

    ```bash
    python test.py --weights runs/train/your_best_model.pt --data config.yaml --img 640
    ```

    This will evaluate the model's performance on the test dataset and generate metrics such as precision, recall, F1 score, and confusion matrices.

3. **View Results**: Test results, including metrics and visual outputs, will be saved to the `runs/test/` directory.

### 3. Running the Detection System

To apply the trained model to a video and detect basketball events:

1. **Prepare Your Input Video**: Ensure the video is in a compatible format (e.g., MP4).
2. **Run the Detection Script**:

    ```bash
    python shot_detector.py --input path_to_video.mp4 --output output_video.mp4 --weights runs/train/your_best_model.pt
    ```

    This command processes the input video, detects key events (such as player movements, basketball location, and shots), and generates an annotated output video.

3. **Analyze the Output**: The annotated video will be saved to the specified output path, showing detected objects and key events throughout the game.

