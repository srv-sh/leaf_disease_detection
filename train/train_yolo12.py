from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/sourav/workplace/leaf_disease_detection/dataset/healthy_vs_disease/data.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO12n model on the 'bus.jpg' image
results = model("/home/sourav/workplace/leaf_disease_detection/dataset/healthy_vs_disease/test/images/frame_1_0017_jpg.rf.3c95bd2deae9de8d76741d571db72733.jpg")