from PIL import Image
import pandas as pd
import os

#removing corrupted images
bad_images = []
for root, _, files in os.walk("./SDNET2018"):
    for f in files:
        if f.endswith(".jpg"):
            path = os.path.join(root,f)
            try:
                Image.open(path).verify()
            except:
                bad_images.append(path)

print("Corrupted images:", len(bad_images))


#Confirm image size consistency
sizes = set()

for root, _, files in os.walk("./SDNET2018"):
    for f in files:
        if f.endswith(".jpg"):
            img = Image.open(os.path.join(root,f))
            sizes.add(img.size)

print(sizes)


#metadata creation
dataset_path = "./SDNET2018"
data = []
for root, _, files in os.walk(dataset_path):
    for f in files:
        if f.lower().endswith(".jpg"):
            class_folder = os.path.basename(root)
            label = 1 if class_folder.startswith("C") else 0

            concrete = os.path.basename(os.path.dirname(root))  # D, P, W

            data.append({
                "image_path": os.path.join(root, f),
                "label": label,
                "concrete": concrete
            })

df = pd.DataFrame(data)

print(df.head())
print(df["label"].value_counts())

# Create output directory
os.makedirs("metadata", exist_ok=True)

# Save full dataset
df.to_csv("metadata/sdnet2018_full.csv", index=False)

print("✅ Saved: metadata/sdnet2018_full.csv")



print("Data cleaning complete!")

