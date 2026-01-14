
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from pathlib import Path
import shutil

# --- Model Definition ---
class ResNet50GenderClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.backbone = models.resnet50(weights=None) 
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# --- Dataset Definition ---
class PA100KGenderDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='val'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.csv_file = self.data_dir / f"{split}.csv"
        
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_file}")
            
        df = pd.read_csv(self.csv_file)
        
        img_col = df.columns[0]
        gender_col = None
        for col in df.columns:
            if 'female' in col.lower() or 'gender' in col.lower():
                gender_col = col
                break
        if not gender_col: 
            gender_col = df.columns[1]

        self.gender_col = gender_col
        img_dir = self.data_dir / "data"

        for _, row in df.iterrows():
            img_name = str(row[img_col])
            img_path = img_dir / img_name
            # Fallback extensions checking skipped for speed, assuming mostly correct from eval
            if not img_path.exists():
                 # Minimal fallback
                 if (img_dir / (Path(img_name).stem + '.jpg')).exists():
                     img_path = img_dir / (Path(img_name).stem + '.jpg')
            
            # Allow even missing files to keep index alignment? No, dataset skips them.
            # But we need to update CSV by filename. So we store filename.
            
            if img_path.exists():
                gender_value = int(row[gender_col])
                # Label mapping: 1 (Female) -> 0, 0 (Male) -> 1
                label = 0 if gender_value == 1 else 1
                self.samples.append((str(img_path), label, img_name))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, img_name = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

def scan_and_fix(model, data_dir, split, threshold=0.99):
    print(f"\nScanning {split} set (Threshold: {threshold})...")
    
    # Setup
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PA100KGenderDataset(data_dir, transform=transform, split=split)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4) # Adjusted BS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    corrections = [] # (filename, new_female_val)
    
    with torch.no_grad():
        for i, (images, labels, filenames) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            confs = torch.max(probs, dim=1).values.cpu().numpy()
            labels = labels.cpu().numpy()
            
            for j in range(len(labels)):
                # label 0 = Female, 1 = Male
                # If label != pred AND conf > threshold
                if preds[j] != labels[j] and confs[j] > threshold:
                    # Logic:
                    # If Pred=1 (Male), Label=0 (Female). We want to set Label to 1 (Male).
                    # CSV 'Female' column: 
                    # Dataset Label 1 (Male) -> Female=0
                    # Dataset Label 0 (Female) -> Female=1
                    
                    # We want to adopt the Pred label.
                    # If Pred=1 (Male) -> New Female=0
                    # If Pred=0 (Female) -> New Female=1
                    
                    new_female_val = 0 if preds[j] == 1 else 1
                    corrections.append((filenames[j], new_female_val))
                    
            if i % 20 == 0:
                print(f"Batch {i}/{len(dataloader)} - Found {len(corrections)} candidates")

    print(f"Total candidates for correction in {split}: {len(corrections)}")
    
    if corrections:
        # Apply corrections
        csv_path = os.path.join(data_dir, f"{split}.csv")
        backup_path = csv_path + ".auto_fix.bak"
        shutil.copy2(csv_path, backup_path)
        print(f"Backed up {split}.csv to {backup_path}")
        
        df = pd.read_csv(csv_path)
        
        # Batch update
        count = 0
        for fname, new_val in corrections:
            # Locate row
            mask = df['Image'] == fname
            if mask.any():
                df.loc[mask, 'Female'] = new_val
                count += 1
                
        df.to_csv(csv_path, index=False)
        print(f"Updated {count} rows in {split}.csv")

def main():
    BASE_DIR = r"d:\Development\CODA\Bareq\VisionTera Project\VisionTera AI"
    MODEL_PATH = os.path.join(BASE_DIR, "infrastructure/models/v_20260107_102312/best_model.pt")
    DATA_DIR = os.path.join(BASE_DIR, "datasets")
    
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50GenderClassifier(num_classes=2)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    
    # Fix Val first (smaller, safer check)
    scan_and_fix(model, DATA_DIR, 'val', threshold=0.90)
    # Fix Train (to solve overfitting)
    scan_and_fix(model, DATA_DIR, 'train', threshold=0.90)

if __name__ == "__main__":
    main()
