import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import YOLOv11  # Assuming you have a YOLOv11 model defined
from dataset import ConcreteCrackDataset  # Custom Dataset for concrete cracks
from evaluator import evaluate_mAP  # Function to evaluate mAP

# Hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 50

# Load Dataset
train_dataset = ConcreteCrackDataset(transform=transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
model = YOLOv11()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if (epoch + 1) % 10 == 0:
        mAP = evaluate_mAP(model, val_loader)  # Placeholder for your evaluation method
        print(f'Epoch [{epoch+1}], mAP: {mAP:.4f}')

# Save the model
torch.save(model.state_dict(), 'yolov11_crack_model.pth')