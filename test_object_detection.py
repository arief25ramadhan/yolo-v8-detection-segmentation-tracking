from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train4/weights/best.pt") 
# Use the model
results = model("media/2007_000272.jpg", save=True)  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format