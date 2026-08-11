import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

from deepfake_detection.models import DeepfakeImageModel
from preprocessing.preprocessor import MediaPreprocessor

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, filepath="models/checkpoint.pth"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss
    }
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer, scheduler=None):
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0, float('inf')
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"Loaded checkpoint from {filepath} (Epoch {checkpoint['epoch']+1})")
    return checkpoint['epoch'] + 1, checkpoint['val_loss']

def plot_metrics(all_labels, all_preds, all_probs, history, output_dir="outputs/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loss & Accuracy curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

def train_model(data_dir, num_epochs=10, batch_size=16, learning_rate=1e-4, resume=False, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Set up preprocessing and custom transforms
    preprocessor = MediaPreprocessor(device=device)

    # Initialize PyTorch Datasets
    if os.path.exists(os.path.join(data_dir, 'real')) and os.path.exists(os.path.join(data_dir, 'fake')):
        import glob
        from sklearn.model_selection import train_test_split
        real_paths = glob.glob(os.path.join(data_dir, 'real', '**', '*.*'), recursive=True)
        fake_paths = glob.glob(os.path.join(data_dir, 'fake', '**', '*.*'), recursive=True)
        
        all_paths = real_paths + fake_paths
        all_labels = [0]*len(real_paths) + [1]*len(fake_paths)
        
        train_p, val_p, train_l, val_l = train_test_split(all_paths, all_labels, test_size=0.2, random_state=42)
        
        class SimpleDataset(Dataset):
            def __init__(self, paths, labels, transform):
                self.paths = paths
                self.labels = labels
                self.transform = transform
            def __len__(self): return len(self.paths)
            def __getitem__(self, idx):
                from PIL import Image
                img = Image.open(self.paths[idx]).convert('RGB')
                return self.transform(img), self.labels[idx]
                
        train_dataset = SimpleDataset(train_p, train_l, preprocessor.train_transforms)
        val_dataset = SimpleDataset(val_p, val_l, preprocessor.val_transforms)
    else:
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print(f"Dataset directory missing or invalid format in {data_dir}.")
            return

        train_dataset = datasets.ImageFolder(train_dir, transform=preprocessor.train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=preprocessor.val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, Loss, Optimizer, Scheduler, EarlyStopping
    model = DeepfakeImageModel(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=5)

    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    checkpoint_path = "models/image_detector_checkpoint.pth"
    if resume and os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Train Loop"):
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Val Loop"):
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_acc = accuracy_score(all_labels, all_preds)

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(epoch_val_loss)
        print(f"Epoch {epoch+1} Summary: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save model if improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "models/image_detector.pth")
            torch.save(model.state_dict(), "models/best_model.pth")
            print("New best model saved to models/image_detector.pth")

        # Save Checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, epoch_val_loss, filepath=checkpoint_path)

        # Early Stopping Check
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

    # Save training plots and metrics
    if len(all_labels) > 0:
        plot_metrics(all_labels, all_preds, all_probs, history)
        
        # Save JSON stats
        with open("outputs/metrics/stats.json", "w") as f:
            json.dump({
                "final_val_accuracy": accuracy_score(all_labels, all_preds),
                "final_val_precision": precision_score(all_labels, all_preds, zero_division=0),
                "final_val_recall": recall_score(all_labels, all_preds, zero_division=0),
                "final_val_f1": f1_score(all_labels, all_preds, zero_division=0),
                "final_val_auc": roc_auc_score(all_labels, all_probs)
            }, f, indent=4)

if __name__ == "__main__":
    vit_path = r"c:\Users\Dell\Downloads\Datasets\VIT_Dataset"
    if os.path.exists(vit_path):
        train_model(data_dir=vit_path, num_epochs=50, batch_size=16, resume=False)
    else:
        train_model(data_dir="datasets/celeb_df_v2", num_epochs=50, resume=False)
