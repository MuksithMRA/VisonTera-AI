from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO('yolo11n-cls.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='gender_dataset_split', epochs=10, imgsz=224, device=0)
    
    # Evaluate
    metrics = model.val()
    
    # Export the model
    success = model.export(format='onnx')

if __name__ == '__main__':
    train()
