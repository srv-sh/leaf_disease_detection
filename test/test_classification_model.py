from ultralytics import YOLO
import cv2
import torch
import os
import yaml
import numpy as np
import timm
import torchvision.transforms as transforms
from PIL import Image

# Load the object detection model
object_detector = YOLO("/home/sourav/workplace/leaf_disease_detection/train/mlartifacts/4/269ca4c6ab0648b18c54281de49312e5/artifacts/weights/best.pt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
object_detector.to(device)

# Load the classification model (choose one: EfficientNet, ViT, or ResNet50)
classifier = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2).to(device)
classifier.load_state_dict(torch.load("/home/sourav/workplace/leaf_disease_detection/train/models/ViT.pth")) # Update with correct path
classifier.eval()

# Define transformations for classification model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class labels
class_labels = {0: "Borer", 1: "Healthy"}

# Define bounding box colors
class_colors = {"Healthy": (255, 0, 0), "Borer": (0, 0, 255)}

# Inference on test images
test_images_folder = "/home/sourav/workplace/leaf_disease_detection/dataset/leaf_detection_dataset/test/images"
output_folder = "results/test_inference"
os.makedirs(output_folder, exist_ok=True)

for image_name in os.listdir(test_images_folder):
    image_path = os.path.join(test_images_folder, image_name)
    print(image_path)
    image = cv2.imread(image_path)
    results = object_detector(image)
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())
            if cls == 0:  # Process only leaf class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                leaf_crop = image[y1:y2, x1:x2]
                
                # Convert crop to PIL and preprocess for classification
                leaf_pil = Image.fromarray(cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2RGB))
                leaf_tensor = transform(leaf_pil).unsqueeze(0).to(device)
                
                # Predict class (Healthy or Borer)
                with torch.no_grad():
                    output = classifier(leaf_tensor)
                    pred_class = torch.argmax(output, dim=1).item()
                    label = class_labels[pred_class]
                    color = class_colors[label]
                
                # Draw bounding box with appropriate color
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save output image
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)

print(f"Inference completed. Results saved in {output_folder}")
