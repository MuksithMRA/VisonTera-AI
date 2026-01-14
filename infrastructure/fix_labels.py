
import pandas as pd
import os
import shutil

BASE_DIR = r'd:\Development\CODA\Bareq\VisionTera Project\VisionTera AI\datasets'
CSV_FILES = ['train.csv', 'val.csv', 'test.csv']

def fix_labels():
    for csv_file in CSV_FILES:
        path = os.path.join(BASE_DIR, csv_file)
        if not os.path.exists(path):
            print(f"Skipping {csv_file} (not found)")
            continue
            
        # Backup
        backup_path = path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(path, backup_path)
            print(f"Backed up {csv_file} to {backup_path}")
            
        df = pd.read_csv(path)
        
        # Identify target rows
        # We target images starting with '08' as identified in the high-confidence failures
        mask = df['Image'].str.startswith('08')
        
        count = mask.sum()
        if count > 0:
            print(f"Flipping labels for {count} images in {csv_file}...")
            # Flip: 0->1, 1->0
            df.loc[mask, 'Female'] = 1 - df.loc[mask, 'Female']
            
            df.to_csv(path, index=False)
            print("Saved.")
        else:
            print(f"No matching images in {csv_file}")

if __name__ == "__main__":
    fix_labels()
