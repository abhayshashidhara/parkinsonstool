import torch
import timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Paths (update if different)
dataset_path = "/kaggle/input/astory"               
weights_path = "/kaggle/working/spiral_model.pth"   

device = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessing 
spiral_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Dataset & loader 
test_dataset = ImageFolder(root=dataset_path, transform=spiral_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
spiral_model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=2).to(device)
state = torch.load(weights_path, map_location=device)
spiral_model.load_state_dict(state)
spiral_model.eval()

# Predictions
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = spiral_model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
conf_matrix = confusion_matrix(all_labels, all_preds)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, zero_division=0))
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
