
import pandas as pd
import os

csv_path = r'd:\Development\CODA\Bareq\VisionTera Project\VisionTera AI\datasets\val.csv'
df = pd.read_csv(csv_path)

filenames = [
    "080461.jpg",
    "080631.jpg",
    "081906.jpg",
    "082382.jpg",
    "080854.jpg"
]

print(f"Checking {len(filenames)} files in {csv_path}...")
found = df[df['Image'].isin(filenames)]
print(found[['Image', 'Female']])
