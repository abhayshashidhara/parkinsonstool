# Evaluate the spiral classifier: quick accuracy 

import os, glob
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

def _looks_like_imagefolder_root(path: str) -> bool:
    # Looks for >=2 subfolders that contain image files
    try:
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    except Exception:
        return False
    if len(subdirs) < 2:
        return False
    for d in subdirs:
        if glob.glob(os.path.join(path, d, "*.*")):
            return True
    return False

def find_dataset_root(preferred: str | None = None) -> str:
    # Guess where the ImageFolder-style dataset lives
    candidates = [preferred, os.environ.get("DATASET_PATH")]
    candidates += [
        "//kaggle/input/first-code-data",
        "/kaggle/input/paper-data",
        "/kaggle/input/pd-imagingdata",
        ".",
    ]
    seen = set()
    for base in candidates:
        if not base or base in seen or not os.path.exists(base):
            continue
        seen.add(base)
        if _looks_like_imagefolder_root(base):
            return base
        for root, dirs, _ in os.walk(base):
            if _looks_like_imagefolder_root(root):
                return root
    raise FileNotFoundError(
        "No ImageFolder root found. Set DATASET_PATH or pass a valid folder."
    )

def find_weights(preferred: str | None = None) -> str:
    # Find the trained weights file
    candidates = [
        preferred,
        os.environ.get("WEIGHTS_PATH"),
        "/kaggle/working/spiral_model.pth",
        "models/spiral_model.pth",
        "spiral_model.pth",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "spiral_model.pth not found. Train first or set WEIGHTS_PATH."
    )

DATASET_ROOT = None
WEIGHTS_PATH = None
BATCH_SIZE = 16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spiral_transform = transforms.Compose([
    # If you trained with grayscale->RGB, add: transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def main():
    dataset_root = find_dataset_root(DATASET_ROOT)
    weights_path = find_weights(WEIGHTS_PATH)

    print(f"Using dataset root: {dataset_root}")
    print(f"Using weights:      {weights_path}")
    print(f"Device:             {device}")

    # Data
    test_dataset = ImageFolder(root=dataset_root, transform=spiral_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=2).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Predictions
    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Quick accuracy
    quick_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"\nQuick Test Accuracy: {quick_acc:.2f}%")

    # Full metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, zero_division=0))
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

if __name__ == "__main__":
    main()
