```markdown
# Crack Analysis Using AI 🛠️

This project detects cracks in concrete, pavements, and walls using deep learning (PyTorch). The dataset used is **SDNET2018**, and the model is trained with **ResNet18**.

---

## Features ✨

- Preprocessing scripts handle **class imbalance** and split datasets.
- PyTorch `Dataset` and `DataLoader` for efficient training.
- Transfer learning using **ResNet18** for binary classification (crack/no crack).
- Weighted loss to handle **dominance of uncracked images (~82%)**.
- Progress tracking using `tqdm`.
- IDE used: **Google Antigravity**.
- Files used: `.py` and `.ipynb`.

---

## Dataset 📂

Download the SDNET2018 dataset from [USU Digital Commons](https://digitalcommons.usu.edu/all_datasets/48/).  

**Folder structure after extraction:**

```

SDNET2018/
├── D
│   ├── CD
│   └── UD
├── P
│   ├── CP
│   └── UP
└── W
├── CW
└── UW

````

- `D` = Doors, `P` = Pavement, `W` = Walls  
- `C` = Cracked, `U` = Uncracked

> **Note:** The dataset is large (~500 MB) and ignored in Git.

---

## Setup 💻

### 1️⃣ Clone the repository

```bash
git clone https://github.com/KritishX/Crack-Analysis-using-AI.git
cd Crack-Analysis-using-AI
````

---

### 2️⃣ Create Python environment (via Anaconda Navigator)

* Open **Anaconda Navigator** → Environments → Create new → Python 3.10
* Activate the environment:

```bash
conda activate CA
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements include:**

```
torch
torchvision
pandas
numpy
scikit-learn
matplotlib
tqdm
Pillow
```

---

### 4️⃣ Dataset setup
(Skip this step if you run the 1_zip_file_extractio.py file which automatically extracts file and saves in a output Dir)
1. Place `SDNET2018.zip` in the project folder.
Extract it:

#### Windows

```powershell
tar -xf SDNET2018.zip
```

#### macOS / Linux

```bash
unzip SDNET2018.zip -d SDNET2018
```

---

### 5️⃣ Preprocessing

```bash
python data_preprocessing.py
```

* Handles **class imbalance**.
* Splits data into **train, validation, and test sets**.
* Saves CSVs and prepares PyTorch loaders.

---

### 6️⃣ Training

```bash
python model_training.py
```

* Uses **ResNet18** pretrained on ImageNet.
* Weighted loss for class imbalance.
* Saves the best model as `best_model.pth`.
* Progress bars and metrics for each epoch.
* Automatically detects **CPU/GPU**.

> Example snippet to see device and dataset info before training:

```python
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Class weights: {class_weights}")
print(f"Using device: {device}")
```

---

### 7️⃣ Evaluation / Inference

```python
from PIL import Image
from torchvision import transforms
import torch
from model_training import model
from data_preprocessing import criterion  # optional

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open("path_to_image.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = img.to(device)
model.to(device)

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1)
    print("Crack Detected" if pred.item() == 1 else "No Crack")
```

---

### Notes for macOS / Linux 🍏🐧

* Use `pip3` if Python 3 is not default.
* For Mac M1/M2 CPU-only:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Project Structure 📁

```
├── 1_zip_file_extraction.py      # Extract dataset
├── 2_data_viewing.ipynb          # Visualize dataset distribution
├── 3_data_cleaning.py            # Optional cleaning
├── 4_data_optimization.py        # Splitting & balancing dataset
├── data_preprocessing.py         # PyTorch Dataset & transforms
├── model_training.py             # Training & evaluation
├── requirements.txt              # Dependencies
└── SDNET2018/                    # Dataset (ignored)
```


```
 

Do you want me to do that too?
```
