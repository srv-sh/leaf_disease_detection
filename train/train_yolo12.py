from ultralytics import YOLO
import mlflow
import os
import torch
import yaml
import numpy as np
import cv2
import random

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:8000")
mlflow.set_experiment("YOLO12 Leaf Detection")

data_yaml_path = "/home/sourav/workplace/leaf_disease_detection/dataset/leaf_detection_dataset/data.yaml"
train_params_list = [
    # Experiment 1: Moderate HSV, moderate flips, and low augmentation
    {"epochs": 300, "batch": 16, "imgsz": 640, "workers": 4, "hsv_h": 0.02, "hsv_s": 0.5, "hsv_v": 0.3, "flipud": 0.5, "fliplr": 0.5, "mosaic": 0.5, "mixup": 0.1, "copy_paste": 0.2},
    
    # Experiment 2: Higher hue, more mosaic augmentation, higher mixup
    {"epochs": 300, "batch": 16, "imgsz": 640, "workers": 4, "hsv_h": 0.05, "hsv_s": 0.6, "hsv_v": 0.4, "flipud": 0.5, "fliplr": 0.5, "mosaic": 0.8, "mixup": 0.3, "copy_paste": 0.1},
    
    # Experiment 3: Strong augmentation with copy-paste, flip variations
    {"epochs": 300, "batch": 16, "imgsz": 640, "workers": 4, "hsv_h": 0.03, "hsv_s": 0.7, "hsv_v": 0.5, "flipud": 0.6, "fliplr": 0.4, "mosaic": 0.9, "mixup": 0.2, "copy_paste": 0.3},
    
    # Experiment 4: Lower saturation and value augmentation, mixup 0.1
    # {"epochs": 300, "batch": 16, "imgsz": 640, "workers": 4, "hsv_h": 0.02, "hsv_s": 0.4, "hsv_v": 0.3, "flipud": 0.5, "fliplr": 0.5, "mosaic": 0.7, "mixup": 0.1, "copy_paste": 0.1},
    
    # # Experiment 5: High mosaic, high flip, lower copy-paste
    # {"epochs": 300, "batch": 16, "imgsz": 640, "workers": 4, "hsv_h": 0.03, "hsv_s": 0.6, "hsv_v": 0.4, "flipud": 0.7, "fliplr": 0.7, "mosaic": 1.0, "mixup": 0.4, "copy_paste": 0.15},
    
    # # Experiment 6: High flip and mixup, low mosaic, moderate copy-paste
    # {"epochs": 300, "batch": 16, "imgsz": 640, "workers": 4, "hsv_h": 0.04, "hsv_s": 0.5, "hsv_v": 0.4, "flipud": 0.7, "fliplr": 0.6, "mosaic": 0.6, "mixup": 0.5, "copy_paste": 0.2},
    
    # # Experiment 7: Very high mosaic, minimal flips, copy-paste 0.3
    # {"epochs": 300, "batch": 16, "imgsz": 640, "workers": 4, "hsv_h": 0.02, "hsv_s": 0.5, "hsv_v": 0.4, "flipud": 0.3, "fliplr": 0.3, "mosaic": 1.0, "mixup": 0.3, "copy_paste": 0.3},
    
    # # Experiment 8: Low hue variation, moderate saturation, flip and mixup high
    # {"epochs": 300, "batch": 16, "imgsz": 640, "workers": 4, "hsv_h": 0.01, "hsv_s": 0.7, "hsv_v": 0.4, "flipud": 0.6, "fliplr": 0.6, "mosaic": 0.7, "mixup": 0.7, "copy_paste": 0.2},
]

best_map50 = 0
best_model_path = ""

for params in train_params_list:
    with mlflow.start_run(nested=True):
        model = YOLO("yolo12l.pt")
        
        mlflow.log_params(params)
        
        results = model.train(
            data=data_yaml_path, 
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            workers=params["workers"],
            hsv_h=params["hsv_h"],
            hsv_s=params["hsv_s"],
            hsv_v=params["hsv_v"],
            flipud=params["flipud"],
            fliplr=params["fliplr"],
            mosaic=params["mosaic"],
            mixup=params["mixup"],
            copy_paste=params["copy_paste"]
        )
        
        last_run_path = model.ckpt_path  # Get the path to the best model
        
        if results.box.map50 > best_map50:
            best_map50 = results.box.map50
            best_model_path = last_run_path

        mlflow.log_metrics({"mAP_50": results.box.map50, "mAP_50_95": results.box.map})
        mlflow.log_artifact(last_run_path)
        print(f"Training run completed with mAP@50: {results.box.map50:.4f}")
        # Ensure the run ends properly
        # mlflow.end_run()

# Load the best model and evaluate
mlflow.log_param("best_model", best_model_path)
model = YOLO(best_model_path)
metrics = model.val(data=data_yaml_path)

mlflow.log_metrics({"Test mAP_50": metrics.box.map50, "Test mAP_50_95": metrics.box.map})
print(f"Best model {best_model_path} evaluated with mAP@50: {metrics.box.map50:.4f}")
