from ultralytics import YOLO
# import torchvision
# import ultralytics
# print(ultralytics.checks())
# print(torchvision.version.cuda)

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="config.yaml", epochs=10, device=0)  # train the model