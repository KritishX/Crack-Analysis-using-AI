"""
Crack Detection Model Training Script
------------------------------------
This script trains a ResNet18 model for binary crack detection on concrete, pavement, and walls.
It uses PyTorch and torchvision for model and data handling.

Features:
- CPU-friendly (Windows-safe with num_workers=0)
- Weighted loss to handle unbalanced classes
- Full training, validation, and test evaluation
- Progress bars using tqdm
- Prints dataset info and class distribution before training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import train_loader, val_loader, test_loader, criterion

def main():
    """
    Main training function. Loads the pretrained ResNet18 model,
    applies a custom binary classification head, trains the model
    with weighted loss, and evaluates on validation and test sets.
    """
    # --------------------------
    # Device configuration
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------
    # Model definition
    # --------------------------
    print("\nInitializing ResNet18 model with pretrained ImageNet weights...")
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification
    model = model.to(device)

    # --------------------------
    # Optimizer
    # --------------------------
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --------------------------
    # Training settings
    # --------------------------
    num_epochs = 5  # Adjust as needed
    best_val_f1 = 0.0

    # --------------------------
    # Dataset info
    # --------------------------
    print("\nDataset Information:")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Class weights used in CrossEntropyLoss: {criterion.weight}\n")

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(num_epochs):
        print(f"\n========== Epoch {epoch+1}/{num_epochs} ==========")
        model.train()
        train_loss = 0
        all_labels, all_preds = [], []

        # Training progress bar
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            train_bar.set_postfix({"loss": loss.item()})

        train_acc = accuracy_score(all_labels, all_preds)
        train_f1  = f1_score(all_labels, all_preds)
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        val_loss = 0
        val_labels, val_preds = [], []
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_bar.set_postfix({"loss": loss.item()})

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1  = f1_score(val_labels, val_preds)
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # --------------------------
        # Save best model
        # --------------------------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model!\n")

    # --------------------------
    # Test evaluation
    # --------------------------
    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_labels, test_preds = [], []
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", leave=False)
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1  = f1_score(test_labels, test_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()
