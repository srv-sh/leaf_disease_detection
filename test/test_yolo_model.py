from ultralytics import YOLO
import cv2
import torch
import os
import yaml
import numpy as np

# Load the trained model
model = YOLO("/home/sourav/workplace/leaf_disease_detection/train/mlartifacts/4/269ca4c6ab0648b18c54281de49312e5/artifacts/weights/best.pt")  # Replace with your checkpoint path

# Set device (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load dataset config from YAML file
data_yaml_path = "/home/sourav/workplace/leaf_disease_detection/dataset/leaf_detection_dataset/data.yaml"  # Replace with your dataset YAML file path
with open(data_yaml_path, 'r') as file:
    data_config = yaml.safe_load(file)

test_images_folder = data_config.get("test", "")

# Create a folder to save results
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Evaluate model on the test dataset
metrics = model.val(data=data_yaml_path)
print(f"Evaluation Results: mAP@50: {metrics.box.map50:.4f}, mAP@50-95: {metrics.box.map:.4f}")

# Ground truth annotations dictionary
gt_annotations = {}

# Load ground truth annotations from label files (YOLO format: class x_center y_center width height)
labels_folder = "/home/sourav/workplace/leaf_disease_detection/dataset/healthy_vs_disease/test/labels"  # Replace with the path to label files
for label_file in os.listdir(labels_folder):
    label_path = os.path.join(labels_folder, label_file)
    image_name = label_file.replace(".txt", ".jpg")  # Assuming images are in JPG format
    with open(label_path, 'r') as f:
        gt_boxes = []
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            cls, x_center, y_center, width, height = parts
            x1 = (x_center - width / 2) * 640  # Assuming 640x640 images
            y1 = (y_center - height / 2) * 640
            x2 = (x_center + width / 2) * 640
            y2 = (y_center + height / 2) * 640
            gt_boxes.append([x1, y1, x2, y2])
        gt_annotations[image_name] = gt_boxes

# Perform inference on test images and save results
output_images_folder = os.path.join(output_folder, "test_inference")
os.makedirs(output_images_folder, exist_ok=True)

# Define class-specific colors
class_colors = {
    0: (0, 0, 255),  # Red for class 0 (healthy)
    1: (255, 0, 0)   # Blue for class 1 (disease)
}

# Function to compute IoU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

iou_values = []

for image_name in os.listdir(test_images_folder):
    image_path = os.path.join(test_images_folder, image_name)
    image = cv2.imread(image_path)
    
    # Perform inference
    results = model(image)
    
    pred_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            
            # Assign predefined color based on class
            color = class_colors.get(cls, (0, 255, 0))  # Default green if unexpected class appears
            
            # Draw bounding box on the image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            pred_boxes.append([x1, y1, x2, y2])
    
    # Compute IoU with ground truth boxes
    if image_name in gt_annotations:
        for pred_box in pred_boxes:
            ious = [compute_iou(pred_box, gt_box) for gt_box in gt_annotations[image_name]]
            if ious:
                iou_values.append(max(ious))  # Store the highest IoU for each prediction
    
    # Save the output image
    output_image_path = os.path.join(output_images_folder, image_name)
    cv2.imwrite(output_image_path, image)

# Compute and display mean IoU
mean_iou = np.mean(iou_values) if iou_values else 0
print(f"Mean IoU on test dataset: {mean_iou:.4f}")

print(f"Inference completed on test set. Results saved in {output_images_folder}")
