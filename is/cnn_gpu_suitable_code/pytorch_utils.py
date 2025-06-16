import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import torch.nn as nn
import torch.nn.functional as F


class XRayDataset(Dataset):
    """
    A custom PyTorch Dataset class for chest X-Ray images.

    This class is designed to handle X-Ray image datasets, supporting both grayscale
    and RGB image modes. It allows for on-the-fly transformations of the images and labels,
    facilitating data augmentation and preprocessing steps.

    Parameters:
    - metadata (pd.DataFrame): DataFrame containing image metadata (e.g., filenames, labels).
    - img_dir (str): Directory path where images are stored.
    - classes (list): List of column names in `metadata` representing the label(s) for each image.
    - img_mode (str, optional): The mode of the images, either "RGB" or "GRAY". Default is "RGB".
    - transform (callable, optional): A function/transform that takes in an image and returns a transformed version. E.g., data augmentation procedures.
    - target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    - filename_prefix (str, optional): Prefix to add to filenames from `metadata` before loading images. Useful if `metadata` filenames do not include a common path prefix that is present in `img_dir`.

    Usage:
    dataset = XRayDataset(metadata=df, img_dir="/path/to/images", classes=['Normal', 'Pneumonia'], img_mode='RGB')
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        img_dir: str,
        classes: list,
        img_mode: str = "RGB",
        transform=None,
        target_transform=None,
    ):
        self.img_metadata = metadata
        self.img_dir = img_dir
        self.classes = classes
        if img_mode.lower() == "rgb":
            self.img_mode = ImageReadMode.RGB
        elif img_mode.lower() == "gray":
            self.img_mode = ImageReadMode.GRAY
        else:
            raise ValueError("Unknown image mode (img_mode), must be GRAY or RGB.")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.img_metadata)

    def __getitem__(self, idx):
        """Retrieves the image and its label at the specified index `idx`.

        Parameters:
        - idx (int): Index of the item to retrieve.

        Returns:
        - tuple: (image, label) where image is the transformed image tensor, and label is the corresponding label tensor.
        """
        # Construct the full path to the image file
        img_path = os.path.join(self.img_dir, self.img_metadata.iloc[idx, 0])

        # Read the image file
        image = read_image(img_path, mode=self.img_mode)

        # Extract label(s) for the current image
        label = torch.tensor(self.img_metadata[self.classes].iloc[idx, :], dtype=torch.float32)

        # Apply transformations to the image and label if any
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


# Code to train the model
def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device
        ) -> float:
    """
    Runs one full pass through the training data and returns the average loss.
    Instead of printing batch losses, we push them into the tqdm progress bar.
    """
    model.train()
    running_loss = 0.0

    # Create a tqdm iterator over the DataLoader
    loop = tqdm(dataloader, desc="Training", unit="batch")

    for batch_idx, (inputs, labels) in enumerate(loop, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Every time through the loop, update the tqdm bar's postfix with the latest loss
        # You can show whichever metric you like; here we send the raw batch loss.
        loop.set_postfix(batch_loss=loss.item())

    # Return average loss over all batches
    avg_loss = running_loss / batch_idx
    return avg_loss


def plot_roc_curves(labels_val, predictions_val, classes, figsize=(8, 8), dpi=150):
    """
    Plot ROC curves for multi-class classification.
    
    Parameters:
    -----------
    labels_val : np.ndarray
        Ground truth binary labels with shape (n_samples, n_classes).
    predictions_val : np.ndarray
        Predicted probabilities with shape (n_samples, n_classes).
    classes : list of str
        List of class names corresponding to each column.
    figsize : tuple, optional
        Size of the figure (default is (8, 8)).
    dpi : int, optional
        Resolution of the figure (default is 150).
    """
    n_classes = len(classes)

    fpr, tpr, roc_auc = {}, {}, []

    # Compute ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_val[:, i], predictions_val[:, i])
        roc_auc.append(auc(fpr[i], tpr[i]))

    # Sort classes by AUC descending
    sorted_by_auc = np.argsort(roc_auc)[::-1]
    color_map = plt.get_cmap("tab20b", n_classes)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for count, i in enumerate(sorted_by_auc):
        ax.plot(
            fpr[i], tpr[i],
            color=color_map.colors[count],
            lw=2,
            label=f"{classes[i]}, area = {roc_auc[i]:.2f}"
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic for Multi-Class Classification')
    ax.legend(loc="lower right", fontsize=10)
    plt.show()
