import os
import logging
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from PIL import Image
import pandas as pd

from app.config import AppConfig

logger = logging.getLogger("VisionTera.Training")

GENDER_CLASSES = ['Female', 'Male']

class PA100KGenderDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        img_dir = self.data_dir / "data"
        
        csv_file = self.data_dir / f"{split}.csv"
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return
        
        if not img_dir.exists():
            logger.error(f"Image directory not found: {img_dir}")
            return
        
        try:
            df = pd.read_csv(csv_file)
            
            img_col = df.columns[0]
            
            gender_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'female' in col_lower or 'gender' in col_lower:
                    gender_col = col
                    break
            
            if gender_col is None:
                gender_col = df.columns[1]
                logger.info(f"Using column '{gender_col}' as gender label")
            
            for _, row in df.iterrows():
                img_name = str(row[img_col])
                img_path = img_dir / img_name
                
                if not img_path.exists():
                    for ext in ['.png', '.jpg', '.jpeg']:
                        test_path = img_dir / (Path(img_name).stem + ext)
                        if test_path.exists():
                            img_path = test_path
                            break
                
                if not img_path.exists():
                    continue
                
                gender_value = int(row[gender_col])
                label = 0 if gender_value == 1 else 1
                
                self.samples.append(str(img_path))
                self.labels.append(label)
            
            male_count = sum(self.labels)
            female_count = len(self.labels) - male_count
            logger.info(f"[{split}] Loaded {len(self.samples)} images. Male: {male_count}, Female: {female_count}")
            
        except Exception as e:
            logger.error(f"Error loading PA-100K CSV: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class ResNet50GenderClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
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


class TrainingPipeline:
    def __init__(self):
        self.is_training = False
        self.status = "idle"
        self.progress = 0.0
        self.data_dir = "datasets"
        self.output_dir = "infrastructure/models"
    
    def run_pipeline(self, epochs=30, batch_size=64):
        if self.is_training:
            logger.warning("Training already in progress")
            return None

        self.is_training = True
        self.status = "initializing"
        
        try:
            logger.info("Starting PA-100K Gender Classification Training (ResNet50)")
            
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            transform_train = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)
            ])
            
            transform_val = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            train_dataset = PA100KGenderDataset(self.data_dir, transform=transform_train, split='train')
            val_dataset = PA100KGenderDataset(self.data_dir, transform=transform_val, split='val')
            
            if len(train_dataset) == 0:
                raise FileNotFoundError("No images loaded from PA-100K train set")
            
            class_counts = [0, 0]
            for label in train_dataset.labels:
                class_counts[label] += 1
            class_weights = [1.0 / max(count, 1) for count in class_counts]
            sample_weights = [class_weights[label] for label in train_dataset.labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            
            num_workers = 4
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                sampler=sampler, 
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size * 2, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            
            logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")
            logger.info(f"Class distribution - Female: {class_counts[0]}, Male: {class_counts[1]}")
            
            self.status = "loading_model"
            
            model = ResNet50GenderClassifier(num_classes=2, pretrained=True)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            logger.info(f"Training on device: {device}")
            logger.info("Using ResNet50 pretrained on ImageNet")
            
            self.status = "training"
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=0.001, 
                epochs=epochs, 
                steps_per_epoch=len(train_loader)
            )
            
            scaler = GradScaler()
            use_amp = torch.cuda.is_available()
            
            best_acc = 0.0
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for i, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    if use_amp:
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    
                    scheduler.step()
                    
                    train_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (predicted == labels).sum().item()
                    train_total += labels.size(0)
                    
                    total_batches = len(train_loader)
                    self.progress = (epoch + (i / total_batches)) / epochs * 100
                    
                    if i % 100 == 0:
                        logger.info(f"Epoch {epoch+1} [{i}/{total_batches}] Loss: {loss.item():.4f}")
                
                train_loss = train_loss / train_total
                train_acc = train_correct / train_total
                
                model.eval()
                val_correct = 0
                val_total = 0
                male_correct = 0
                male_total = 0
                female_correct = 0
                female_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        
                        if use_amp:
                            with autocast():
                                outputs = model(images)
                        else:
                            outputs = model(images)
                        
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)
                        
                        male_mask = labels == 1
                        female_mask = labels == 0
                        male_correct += (predicted[male_mask] == labels[male_mask]).sum().item()
                        male_total += male_mask.sum().item()
                        female_correct += (predicted[female_mask] == labels[female_mask]).sum().item()
                        female_total += female_mask.sum().item()
                
                val_acc = val_correct / val_total if val_total > 0 else 0
                male_acc = male_correct / male_total if male_total > 0 else 0
                female_acc = female_correct / female_total if female_total > 0 else 0
                
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                logger.info(f"  Male Acc: {male_acc:.4f} | Female Acc: {female_acc:.4f}")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    target_path = Path(self.output_dir) / "resnet50-gender.pt"
                    torch.save(model.state_dict(), target_path)
                    logger.info(f"Saved best model to {target_path} (Acc: {val_acc:.4f})")

            self.status = "completed"
            self.progress = 100.0
            logger.info(f"Training completed! Best accuracy: {best_acc:.4f}")
            return str(Path(self.output_dir) / "resnet50-gender.pt")

        except Exception as e:
            self.status = "failed"
            logger.error(f"Training pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            self.is_training = False


pipeline = TrainingPipeline()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    result = pipeline.run_pipeline(epochs=args.epochs, batch_size=args.batch_size)
    if result:
        print(f"Training completed. Model saved to: {result}")
    else:
        print("Training failed.")
