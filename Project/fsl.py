import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import random

# Define dataset path
DATASET_PATH = r"D:\Gitesh\New folder\!Normal & Abnormal Model"

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Custom dataset for Siamese Network with Normal/Abnormal + Body Part Classification
class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.normal_images = []
        self.abnormal_images = []
        self.image_labels = []

        # Check if directories exist
        normal_path = os.path.join(root_dir, "Normal")
        abnormal_path = os.path.join(root_dir, "Abnormality")

        if not os.path.exists(normal_path) or not os.path.exists(abnormal_path):
            raise ValueError(f"Dataset folders not found: {normal_path}, {abnormal_path}")

        # Load normal images with body parts
        for category in os.listdir(normal_path):
            category_path = os.path.join(normal_path, category)
            if os.path.isdir(category_path):
                for img in os.listdir(category_path):
                    self.normal_images.append(os.path.join(category_path, img))
                    self.image_labels.append((0, category))  # Normal = 0

        # Load abnormal images with body parts
        for category in os.listdir(abnormal_path):
            category_path = os.path.join(abnormal_path, category)
            if os.path.isdir(category_path):
                for img in os.listdir(category_path):
                    self.abnormal_images.append(os.path.join(category_path, img))
                    self.image_labels.append((1, category))  # Abnormal = 1

        self.total_images = self.normal_images + self.abnormal_images

        if len(self.total_images) == 0:
            raise ValueError("No images found in dataset folders!")

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img1_path, (label1, body_part1) = self.total_images[idx], self.image_labels[idx]
        is_same_class = random.randint(0, 1)  # 50% same class, 50% different class

        if label1 == 0:  # Normal
            img2_path, (label2, body_part2) = random.choice(
                list(zip(self.normal_images, self.image_labels)) if is_same_class else
                list(zip(self.abnormal_images, self.image_labels))
            )
        else:  # Abnormal
            img2_path, (label2, body_part2) = random.choice(
                list(zip(self.abnormal_images, self.image_labels)) if is_same_class else
                list(zip(self.normal_images, self.image_labels))
            )

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor([1.0 if is_same_class else 0.0], dtype=torch.float32)

        # Convert body part categories to numerical labels
        body_parts = list(set([b for _, b in self.image_labels]))
        body_part1_idx = body_parts.index(body_part1)
        body_part2_idx = body_parts.index(body_part2)

        return img1, img2, label, torch.tensor(label1), torch.tensor(body_part1_idx)

# Load dataset
dataset = SiameseDataset(DATASET_PATH, transform=transform)

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=0)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16, num_workers=0)

# Define Siamese Network for Few-Shot Learning with Normal/Abnormal + Body Part Classification
class SiameseNetwork(nn.Module):
    def __init__(self, num_body_parts):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Output layers
        self.normal_abnormal_classifier = nn.Linear(128, 2)  # Normal/Abnormal Classification
        self.body_part_classifier = nn.Linear(128, num_body_parts)  # Body Part Classification

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        normal_abnormal_pred = self.normal_abnormal_classifier(features)
        body_part_pred = self.body_part_classifier(features)
        return features, normal_abnormal_pred, body_part_pred

# Define Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(label * torch.pow(euclidean_distance, 2) + 
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_body_parts = len(set([b for _, b in dataset.image_labels]))
model = SiameseNetwork(num_body_parts).to(device)

contrastive_loss_fn = ContrastiveLoss()
classification_loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for img1, img2, label, normal_abnormal_label, body_part_label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        normal_abnormal_label, body_part_label = normal_abnormal_label.to(device), body_part_label.to(device)

        optimizer.zero_grad()

        output1, normal_abnormal_pred1, body_part_pred1 = model(img1)
        output2, normal_abnormal_pred2, body_part_pred2 = model(img2)

        loss = (contrastive_loss_fn(output1, output2, label) +
                classification_loss_fn(normal_abnormal_pred1, normal_abnormal_label) +
                classification_loss_fn(body_part_pred1, body_part_label))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Accuracy Calculation
        _, predicted_labels = torch.max(normal_abnormal_pred1, 1)
        total_correct += (predicted_labels == normal_abnormal_label).sum().item()
        total_samples += normal_abnormal_label.size(0)

    accuracy = total_correct / total_samples * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save Model
torch.save(model.state_dict(), "siamese_model.pth")
print("Model Training Complete and Saved!")
