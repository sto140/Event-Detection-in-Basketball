from ultralytics import YOLO

# Load a model
model = YOLO('C:/Users/tolu1/CE301 basketball Object/Large Model.pt')  # load a custom model

#use model to predict on images in the test file
source = 'C:/Users/tolu1/Basketball Object Detection.v9i.yolov8/test/images'

# only detect when confidence is above 0.5 and save detections to a file
model.predict(source, save=True, imgsz=640, conf=0.5)
