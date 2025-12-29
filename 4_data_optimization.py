import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset CSV
df = pd.read_csv("metadata/sdnet2018_full.csv")

print("Total samples:", len(df))
print(df["label"].value_counts())
print("Data loading completed!")

#We’ll use 70% train, 15% validation, 15% test, stratified by label to preserve the crack/uncrack ratio
# Train / Temp split
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,            # 30% goes to val+test
    stratify=df["label"],     # maintain class ratio
    random_state=42
)

# Validation / Test split
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,            # split temp into 50/50 → 15% each
    stratify=temp_df["label"],
    random_state=42
)

print("Train:", train_df.shape, train_df["label"].value_counts())
print("Validation:", val_df.shape, val_df["label"].value_counts())
print("Test:", test_df.shape, test_df["label"].value_counts())


