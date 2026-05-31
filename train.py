import os
import pandas as pd
import torch

print("=== GPU INFO ===")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\n=== DATASETS ===")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print("\n=== LOADING CSV ===")

csv_path = "/kaggle/input/datasets/abderrahim261/accidents-severity/caract-2024.csv"

df = pd.read_csv(csv_path,sep=";")

print("\nDataset shape:")
print(df.shape)

print("\nFirst rows:")
print(df.head())