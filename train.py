from ultralytics import YOLO



# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights coco dataset
model = YOLO('Large Model.pt.pt') # resume training from weights of previous training (100 epoch)


# Use the model
# model.train(data="config.yaml", epochs=3)  # train the model
model.train(data="config.yaml", epochs=100, patience=100, imgsz=640, optimizer='auto' )  # train the model using new dataset with additional classes for madebasket
