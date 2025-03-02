import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
import mlflow
import mlflow.pytorch
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# Configure MLflow
mlflow.set_tracking_uri("http://0.0.0.0:8000")  # Ensure MLflow server is running
mlflow.set_experiment("Leaf Disease Detection")


# Define dataset paths
data_dir = "/home/sourav/workplace/leaf_disease_detection/dataset/cropped_data" 
train_dir = f"{data_dir}/train"
valid_dir = f"{data_dir}/valid"
test_dir = f"{data_dir}/test"

# Hyperparameters
batch_size = 64
epochs = 50
learning_rate = 0.001
num_classes = len(os.listdir(train_dir))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load models
efficientnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes).to(device)
vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes).to(device)
resnet = timm.create_model("resnet50", pretrained=True, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_eff = optim.Adam(efficientnet.parameters(), lr=learning_rate)
optimizer_vit = optim.Adam(vit.parameters(), lr=learning_rate)
optimizer_resnet = optim.Adam(resnet.parameters(), lr=learning_rate)

# Training function
def train_model(model, optimizer, train_loader, valid_loader, model_name):
    print(f"Training {model_name}...")
    with mlflow.start_run(nested=True):
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "model": model_name})
        model.train()
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            all_labels, all_preds = [], []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
            
            val_acc, precision, recall, f1, cm = evaluate_model(model, valid_loader)
            mlflow.log_metrics({"train_loss": total_loss/len(train_loader), "train_acc": correct/total, "valid_acc": val_acc, 
                                "precision": precision, "recall": recall, "f1_score": f1})
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {correct/total:.4f}, Valid Acc: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        
        # Save the trained model locally
        model_save_path = f"./models/{model_name}.pth"
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")
        
        # Log model with input example and signature
        example_input = torch.randn(1, 3, 224, 224).cpu().numpy()
        mlflow.pytorch.log_model(model, model_name, input_example=example_input)
        print(f"{model_name} Training Complete!\n")

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    mlflow.log_metrics({"precision": precision, "recall": recall, "f1_score": f1})
    return correct / total, precision, recall, f1, cm

# Train models
train_model(efficientnet, optimizer_eff, train_loader, valid_loader, "EfficientNet")
train_model(vit, optimizer_vit, train_loader, valid_loader, "ViT")
train_model(resnet, optimizer_resnet, train_loader, valid_loader, "ResNet50")

# Test models
def test_model(model, test_loader, model_name):
    accuracy, precision, recall, f1, cm = evaluate_model(model, test_loader)
    mlflow.log_metrics({f"{model_name}_test_acc": accuracy, f"{model_name}_precision": precision, f"{model_name}_recall": recall, f"{model_name}_f1_score": f1})
    print(f"{model_name} Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

test_model(efficientnet, test_loader, "EfficientNet")
test_model(vit, test_loader, "ViT")
test_model(resnet, test_loader, "ResNet50")

print("Training complete. Run 'mlflow ui' and open http://localhost:5000 to view results.")
