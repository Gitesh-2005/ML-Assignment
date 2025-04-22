import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define the Siamese Model with matching classifier names
class SiameseFSLModel(nn.Module):
    def __init__(self):
        super(SiameseFSLModel, self).__init__()

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

        # Adjust classifier names to match saved model
        self.normal_abnormal_classifier = nn.Linear(128, 2)  # Normal & Abnormal
        self.body_part_classifier = nn.Linear(128, 12)  # Adjust for number of body parts if needed

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        out_normal_abnormal = self.normal_abnormal_classifier(x)
        out_body_part = self.body_part_classifier(x)

        return out_normal_abnormal, out_body_part

# Define Transformation (Must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseFSLModel().to(device)

# Load trained weights with `strict=False` to ignore mismatches
state_dict = torch.load(r"D:\Gitesh\New folder\!Normal & Abnormal Model\siamese_model.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)  # Allows ignoring missing/extra keys

model.eval()
print("Model loaded successfully!")

# Inference Function
def predict_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_normal_abnormal, _ = model(img)  # We only need normal/abnormal classification
        predicted_class = torch.argmax(out_normal_abnormal, dim=1).item()

    label_map = {0: "Normal", 1: "Abnormal"}
    return label_map[predicted_class]

# Example: Classify a new image
test_img_path = r"D:\Gitesh\New folder\!Normal & Abnormal Model\testing purpose images\Normal Maternal Cervix\Patient00248_Plane4_1_of_1.png"
predicted_label = predict_image(test_img_path)
print(f"Predicted Label: {predicted_label}")
