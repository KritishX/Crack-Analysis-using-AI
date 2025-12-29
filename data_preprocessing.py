import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from collections import Counter


# Paths to your saved CSVs from 4_data_optimization.py
train_csv = "metadata/train.csv"
val_csv   = "metadata/val.csv"
test_csv  = "metadata/test.csv"

train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
test_df  = pd.read_csv(test_csv)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


#let's start data preprocessing

#We’ll prepare PyTorch transforms:
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#Create PyTorch Dataset Class
class CrackDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = self.df.loc[idx, "label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Assuming train_df, val_df, test_df are already loaded
train_dataset = CrackDataset(train_df, train_transform)
val_dataset   = CrackDataset(val_df, test_transform)
test_dataset  = CrackDataset(test_df, test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

#Handle Class Imbalance
#Since uncracked dominate (~82%), we use weighted loss:
label_counts = Counter(train_df["label"])
total = sum(label_counts.values())

class_weights = [total / label_counts[0], total / label_counts[1]]
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights)

#Use in PyTorch CrossEntropyLoss:
criterion = nn.CrossEntropyLoss(weight=class_weights)