import torch, torch.nn as nn, torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/kaggle/input/pd-imagingdata/PD_imagingData"

spiral_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

dataset = ImageFolder(root=dataset_path, transform=spiral_transform)
print("class_to_idx:", dataset.class_to_idx)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "spiral_model.pth")
print("Spiral classifier trained and saved.")
spiral_model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=2).to(DEVICE)
spiral_model.load_state_dict(torch.load("/kaggle/working/spiral_model.pth", map_location=DEVICE))
spiral_model.eval()

# Predict helper
def predict_spiral_image(image_path):
    img = Image.open(image_path).convert("RGB")
    x = spiral_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = spiral_model(x)
        pred = torch.argmax(out, dim=1).item()
    return ("The spiral test suggests signs of Parkinson’s."
            if pred == 1 else
            "The spiral test does not show signs of Parkinson’s.")
