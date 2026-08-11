"""
train_attributes.py
====================
Trains the MultiAttributeFaceModel on all available datasets:
  - Gender (0=Male, 1=Female) + Skin Tone (0-4) → UTKFace  (archive 3)
  - Hair Texture (5 classes)                     → Hair Texture dataset (archive 4)

The shared MobileNetV3-Small backbone is updated by both loss signals
simultaneously — each batch alternates between the two datasets so
every training step teaches the backbone something useful.
"""

import os, glob, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from models.attribute_models import MultiAttributeFaceModel

# ─── PATHS ─────────────────────────────────────────────────────────────────────
UTK_DIR  = r"c:\Users\Dell\Downloads\Datasets\archive (3)"
HAIR_DIR = r"c:\Users\Dell\Downloads\Datasets\archive (4)"
MODEL_OUT = "models/best_attribute_model.pth"

# Hair texture class order must match MultiAttributeFaceModel's hair_head output
HAIR_CLASSES = ['Straight', 'Wavy', 'curly', 'dreadlocks', 'kinky']

# Skin-tone / race label mapping (UTKFace: 0=White,1=Black,2=Asian,3=Indian,4=Other)
SKIN_CLASSES = 5

# ─── AUGMENTATION TRANSFORMS ──────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ─── UTKFace DATASET (Gender + Skin Tone) ─────────────────────────────────────
class UTKFaceDataset(Dataset):
    """Reads UTKFace images and parses age_gender_race from filename."""
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        fname = os.path.basename(path).split('.')[0]  # strip .jpg.chip.jpg etc.
        parts = fname.split('_')
        gender, race = 0, 0
        try:
            gender = int(parts[1])          # 0=Male, 1=Female
            race   = min(max(int(parts[2]), 0), SKIN_CLASSES - 1)
        except (IndexError, ValueError):
            pass
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(gender, dtype=torch.float32), torch.tensor(race, dtype=torch.long)


# ─── Hair Texture DATASET ─────────────────────────────────────────────────────
class HairTextureDataset(Dataset):
    """Reads hair texture images from class-named subdirectories."""
    def __init__(self, items, transform=None):
        # items: list of (path, class_idx)
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ─── DATA LOADERS ─────────────────────────────────────────────────────────────
def get_utk_loaders(batch_size=32):
    paths = glob.glob(os.path.join(UTK_DIR, '**', '*.jpg*'), recursive=True)
    paths = [p for p in paths if len(os.path.basename(p).split('_')) >= 3]
    if not paths:
        raise FileNotFoundError(f"No UTKFace images found in {UTK_DIR}")
    tr, va = train_test_split(paths, test_size=0.2, random_state=42)
    print(f"UTKFace  — Train: {len(tr)}, Val: {len(va)}")
    return (
        DataLoader(UTKFaceDataset(tr, train_tf), batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(UTKFaceDataset(va, val_tf),   batch_size=batch_size, shuffle=False, num_workers=0),
    )


def get_hair_loaders(batch_size=32):
    items = []
    for cls_idx, cls_name in enumerate(HAIR_CLASSES):
        for ext in ('jpg', 'jpeg', 'png'):
            for p in glob.glob(os.path.join(HAIR_DIR, '**', cls_name, f'*.{ext}'), recursive=True):
                items.append((p, cls_idx))
            for p in glob.glob(os.path.join(HAIR_DIR, cls_name, f'*.{ext}')):
                items.append((p, cls_idx))
    # deduplicate
    items = list(set(items))
    if not items:
        raise FileNotFoundError(f"No hair texture images found in {HAIR_DIR}")
    random.shuffle(items)
    split = int(0.8 * len(items))
    tr_items, va_items = items[:split], items[split:]
    print(f"HairTex  — Train: {len(tr_items)}, Val: {len(va_items)}")
    # Print class distribution
    from collections import Counter
    dist = Counter(label for _, label in tr_items)
    for ci, cn in enumerate(HAIR_CLASSES):
        print(f"  {cn}: {dist.get(ci, 0)} train images")
    return (
        DataLoader(HairTextureDataset(tr_items, train_tf), batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(HairTextureDataset(va_items, val_tf),   batch_size=batch_size, shuffle=False, num_workers=0),
    )


# ─── TRAINING ─────────────────────────────────────────────────────────────────
def train_attributes(epochs=50, lr=3e-4, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining Face Attribute Models on device: {device}")
    print(f"Epochs: {epochs}  |  LR: {lr}  |  Batch: {batch_size}\n")

    utk_train, utk_val   = get_utk_loaders(batch_size)
    hair_train, hair_val = get_hair_loaders(batch_size)

    model = MultiAttributeFaceModel(pretrained=True).to(device)

    # Load previous checkpoint if it exists (resume training)
    os.makedirs("models", exist_ok=True)
    start_epoch = 0
    if os.path.exists(MODEL_OUT):
        try:
            model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
            print(f"Resumed from existing checkpoint: {MODEL_OUT}")
        except Exception as e:
            print(f"Could not resume: {e} — starting fresh")

    gender_crit = nn.BCEWithLogitsLoss()
    skin_crit   = nn.CrossEntropyLoss()
    hair_crit   = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        # ── UTKFace pass (Gender + Skin Tone) ──────────────────────────────
        utk_loss_sum = 0.0
        utk_iter = iter(utk_train)
        for imgs, genders, races in tqdm(utk_iter, desc=f"Epoch {epoch+1}/{epochs} [UTK]"):
            imgs    = imgs.to(device)
            genders = genders.unsqueeze(1).to(device)
            races   = races.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = gender_crit(out['gender'], genders) + skin_crit(out['skin_tone'], races)
            loss.backward()
            optimizer.step()
            utk_loss_sum += loss.item()

        # ── Hair Texture pass ───────────────────────────────────────────────
        hair_loss_sum = 0.0
        hair_iter = iter(hair_train)
        for imgs, labels in tqdm(hair_iter, desc=f"Epoch {epoch+1}/{epochs} [Hair]"):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = hair_crit(out['hair_texture'], labels)
            loss.backward()
            optimizer.step()
            hair_loss_sum += loss.item()

        scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        g_preds, g_labels = [], []
        s_preds, s_labels = [], []
        h_preds, h_labels = [], []

        with torch.no_grad():
            # UTKFace val
            for imgs, genders, races in utk_val:
                imgs    = imgs.to(device)
                genders = genders.unsqueeze(1).to(device)
                races   = races.to(device)
                out = model(imgs)
                val_loss += (gender_crit(out['gender'], genders) + skin_crit(out['skin_tone'], races)).item()

                g_prob = torch.sigmoid(out['gender']).squeeze(1)
                g_preds.extend((g_prob > 0.5).long().cpu().numpy())
                g_labels.extend(genders.squeeze(1).long().cpu().numpy())

                s_preds.extend(torch.argmax(out['skin_tone'], 1).cpu().numpy())
                s_labels.extend(races.cpu().numpy())

            # Hair val
            for imgs, labels in hair_val:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                out = model(imgs)
                val_loss += hair_crit(out['hair_texture'], labels).item()

                h_preds.extend(torch.argmax(out['hair_texture'], 1).cpu().numpy())
                h_labels.extend(labels.cpu().numpy())

        g_acc = accuracy_score(g_labels, g_preds) * 100
        s_acc = accuracy_score(s_labels, s_preds) * 100
        h_acc = accuracy_score(h_labels, h_preds) * 100

        print(
            f"\nEpoch {epoch+1:02d}/{epochs} | Val Loss: {val_loss:.4f} | "
            f"Gender Acc: {g_acc:.1f}% | Skin Acc: {s_acc:.1f}% | Hair Acc: {h_acc:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  [SAVED] Best model -> {MODEL_OUT}  (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best model saved to {MODEL_OUT}")


if __name__ == "__main__":
    train_attributes(epochs=50, lr=3e-4, batch_size=32)
