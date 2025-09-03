# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from PIL import Image

# Path to the dataset (update if needed)
dataset_path = "/kaggle/input/astory"

# Define how images will be preprocessed before training
spiral_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load dataset and prepare batches
dataset = ImageFolder(root=dataset_path, transform=spiral_transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create the model (ResNet18 with 2 output classes)
model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=2)
model = model.cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):  # run for 5 epochs
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "spiral_model.pth")
print(" Spiral classifier trained & saved.")

# Load the model for prediction
spiral_model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=2)
spiral_model.load_state_dict(torch.load("/kaggle/working/spiral_model.pth"))
spiral_model.eval().cuda()

# Function to predict from a spiral image
def predict_spiral_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = spiral_transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        out = spiral_model(img_tensor)
        pred = torch.argmax(out, dim=1).item()
    return "The spiral test suggests **signs of Parkinson’s**." if pred == 1 else "The spiral test **does not show signs of Parkinson’s**."
