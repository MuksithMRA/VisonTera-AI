import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import numpy as np
import os
from pathlib import Path

# --- Model Definition (Must match training) ---
class ResNet50GenderClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        # weights=None because we load custom weights
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
        self.labels = []
        
        img_dir = self.data_dir / "data"
        csv_file = self.data_dir / f"{split}.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV not found: {csv_file}")
            
        df = pd.read_csv(csv_file)
        
        # Identify columns (simplified from training logic for robustness)
        img_col = df.columns[0]
        gender_col = None
        for col in df.columns:
            if 'female' in col.lower() or 'gender' in col.lower():
                gender_col = col
                break
        if not gender_col: 
            gender_col = df.columns[1]

        for _, row in df.iterrows():
            img_name = str(row[img_col])
            img_path = img_dir / img_name
            # Fallback extensions
            if not img_path.exists():
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = img_dir / (Path(img_name).stem + ext)
                    if test_path.exists():
                        img_path = test_path
                        break
            
            if img_path.exists():
                gender_value = int(row[gender_col])
                # Mapping from training: 1 (Female) -> 0, 0 (Male) -> 1
                label = 0 if gender_value == 1 else 1
                self.samples.append(str(img_path))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

def evaluate(model_path, data_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    model = ResNet50GenderClassifier(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Data Loader
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = PA100KGenderDataset(data_dir, transform=transform_val, split='val')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 3. Inference
    all_labels = []
    all_preds = []
    all_probs = [] # Probabilities for positive class (Male=1)
    failures = [] # (prob_male, true_label, img_path)

    print(f"Evaluating on {len(dataset)} images...")
    with torch.no_grad():
        for i, (images, labels, paths) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Class 1 is Male
            probs_male = probabilities[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.numpy()
            
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs_male)

            # Collect failures for "Worst Predictions"
            for j in range(len(labels)):
                is_wrong = preds[j] != labels[j]
                confidence = probabilities[j, preds[j]].item()
                if is_wrong:
                    failures.append({
                        'path': paths[j],
                        'true': labels[j],
                        'pred': preds[j],
                        'conf': confidence,
                        'prob_male': probs_male[j]
                    })
            
            if i % 50 == 0:
                print(f"Batch {i}/{len(dataloader)}")

    # 4. Generate Visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Female (0)', 'Male (1)'], 
                yticklabels=['Female (0)', 'Male (1)'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # B. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # C. Confidence Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(all_probs, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Distribution of Predicted Probabilities (Male Class)')
    plt.xlabel('Probability of being Male')
    plt.ylabel('Count')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'confidence_histogram.png'))
    plt.close()

    # D. Worst Predictions Gallery
    # Sort by confidence (high confidence but wrong)
    failures.sort(key=lambda x: x['conf'], reverse=True)
    top_failures = failures[:16]
    
    if top_failures:
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for idx, ax in enumerate(axes.flat):
            if idx < len(top_failures):
                item = top_failures[idx]
                img = Image.open(item['path'])
                label_map = {0: 'Female', 1: 'Male'}
                ax.imshow(img)
                ax.set_title(f"T:{label_map[item['true']]} P:{label_map[item['pred']]}\nConf:{item['conf']:.2f}", color='red')
                ax.axis('off')
            else:
                ax.axis('off')
        plt.suptitle("Top High-Confidence Errors (Worst Predictions)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'worst_predictions.png'))
        plt.close()

        # Debug: Print top failures paths and save to file
        print("\nTop High-Confidence Failures:")
        with open(os.path.join(output_dir, 'failures.txt'), 'w', encoding='utf-8') as f:
            for fail in top_failures:
                line = f"File: {fail['path']}, Truth: {fail['true']}, Pred: {fail['pred']}, Conf: {fail['conf']:.4f}"
                print(line)
                f.write(line + "\n")

    # E. Print Report
    report = classification_report(all_labels, all_preds, target_names=['Female', 'Male'])
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(output_dir, 'detailed_metrics.txt'), 'w') as f:
        f.write(report)
        f.write(f"\nROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    BASE_DIR = r"d:\Development\CODA\Bareq\VisionTera Project\VisionTera AI"
    MODEL_PATH = os.path.join(BASE_DIR, "infrastructure/models/v_20260107_102312/best_model.pt")
    DATA_DIR = os.path.join(BASE_DIR, "datasets")
    OUTPUT_DIR = os.path.join(BASE_DIR, "infrastructure/models/v_20260107_102312/evaluation")
    
    evaluate(MODEL_PATH, DATA_DIR, OUTPUT_DIR)
