import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

from pytorch_utils import XRayDataset, train_model

# ------------------------------------------------------------------------------
# Load Metadata & Prepare Paths
# ------------------------------------------------------------------------------

path_data = "../datasets/ChestX_subset_small/"
path_images = os.path.join(path_data, "images_low_resolution")
metadata_df = pd.read_csv(os.path.join(path_data, "metadata.csv"))

OUTPUT_PATH = "./nutzer1"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ------------------------------------------------------------------------------
# Define Labels and Classes
# ------------------------------------------------------------------------------

non_class_columns = [
    'image', 'follow_up_no', 'patient_id', 'patient_age', 'gender', 'view_position'
]
classes = [c for c in metadata_df.columns if c not in non_class_columns]
num_classes = len(classes)

# ------------------------------------------------------------------------------
# Train/Val/Test Split
# ------------------------------------------------------------------------------

train_df, test_df = train_test_split(
    metadata_df, test_size=0.15, stratify=metadata_df['atelectasis'], random_state=0
)
test_df, val_df = train_test_split(
    test_df, test_size=0.5, stratify=test_df['atelectasis'], random_state=0
)

print(20 * ">", " Data Split ", 20 * "<")
print(f"Training set size: {train_df.shape}")
print(f"Validation set size: {val_df.shape}")
print(f"Test set size: {test_df.shape}")

# ------------------------------------------------------------------------------
# Define Transforms & Dataloaders
# ------------------------------------------------------------------------------

transforms = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=3),  # <-- ResNet expects 3 channels
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

training_data = XRayDataset(train_df, path_images, classes=classes, transform=transforms)
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)

validation_data = XRayDataset(val_df, path_images, classes=classes, transform=transforms)
validation_dataloader = DataLoader(validation_data, batch_size=16, shuffle=True)

# ------------------------------------------------------------------------------
# Select Device
# ------------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------------------
# Initialize Transfer Learning Model (ResNet18)
# ------------------------------------------------------------------------------

model = models.resnet18(weights='IMAGENET1K_V1')
# optionally, you can freeze the layers
FREEZE_LAYERS = False  # Set to True if you want to freeze the layers
if FREEZE_LAYERS:
    print("Freezing layers of the model.")
    for param in model.parameters():
        param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

print(f"Transfer learning model initialized with {num_classes} classes.")

# ------------------------------------------------------------------------------
# Loss, Optimizer, Scheduler
# ------------------------------------------------------------------------------

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

EPOCHS = 10
history = train_model(
    model=model,
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    epochs=EPOCHS,
    output_path=OUTPUT_PATH,
    filename_base="resnet18_transfer_learning",
    verbose=True
)
