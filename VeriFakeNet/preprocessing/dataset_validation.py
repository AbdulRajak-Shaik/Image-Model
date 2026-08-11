import os
import glob
from PIL import Image

def audit_datasets(dataset_base_dir=r"c:\Users\Dell\Downloads\Datasets"):
    print("=== STARTING DATASET AUDIT ===")
    results = {}
    
    # 1. VIT_Dataset (Authenticity: Real vs Fake)
    vit_dir = os.path.join(dataset_base_dir, "VIT_Dataset")
    if os.path.exists(vit_dir):
        real_imgs = glob.glob(os.path.join(vit_dir, "real", "**", "*.*"), recursive=True)
        fake_imgs = glob.glob(os.path.join(vit_dir, "fake", "**", "*.*"), recursive=True)
        results["VIT_Dataset"] = {
            "task": "Authenticity (Real / Fake)",
            "real_count": len(real_imgs),
            "fake_count": len(fake_imgs),
            "total": len(real_imgs) + len(fake_imgs)
        }
        print(f"VIT_Dataset: {len(real_imgs)} Real, {len(fake_imgs)} Fake (Total: {len(real_imgs) + len(fake_imgs)})")

    # 2. Archive (3) - UTKFace (Gender & Skin Tone)
    utk_dir = os.path.join(dataset_base_dir, "archive (3)")
    if os.path.exists(utk_dir):
        utk_images = glob.glob(os.path.join(utk_dir, "**", "*.jpg*"), recursive=True)
        gender_counts = {0: 0, 1: 0} # 0: Male, 1: Female
        race_counts = {} # 0: White, 1: Black, 2: Asian, 3: Indian, 4: Others
        valid_utk = 0
        corrupt_utk = 0
        
        for img_path in utk_images:
            fname = os.path.basename(img_path)
            parts = fname.split('_')
            if len(parts) >= 3:
                try:
                    age = int(parts[0])
                    gender = int(parts[1])
                    race = int(parts[2])
                    gender_counts[gender] = gender_counts.get(gender, 0) + 1
                    race_counts[race] = race_counts.get(race, 0) + 1
                    valid_utk += 1
                except ValueError:
                    corrupt_utk += 1
        results["archive (3) UTKFace"] = {
            "task": "Gender & Skin Tone / Race",
            "total_images": len(utk_images),
            "parsed": valid_utk,
            "gender_counts": gender_counts,
            "race_counts": race_counts
        }
        print(f"UTKFace Archive (3): {valid_utk} parsed images. Gender: {gender_counts}, Race: {race_counts}")

    # 3. Archive (4) - Hair Texture
    hair_dir = os.path.join(dataset_base_dir, "archive (4)", "data")
    if os.path.exists(hair_dir):
        textures = {}
        for class_name in os.listdir(hair_dir):
            class_path = os.path.join(hair_dir, class_name)
            if os.path.isdir(class_path):
                imgs = glob.glob(os.path.join(class_path, "*.*"))
                textures[class_name] = len(imgs)
        results["archive (4) Hair Texture"] = {
            "task": "Hair Texture",
            "categories": textures
        }
        print(f"Hair Texture Archive (4): {textures}")

    # 4. Archive (2) - FER Expressions
    fer_dir = os.path.join(dataset_base_dir, "archive (2)")
    if os.path.exists(fer_dir):
        exp_counts = {}
        for split in ['train', 'test']:
            split_path = os.path.join(fer_dir, split)
            if os.path.exists(split_path):
                for cat in os.listdir(split_path):
                    cat_path = os.path.join(split_path, cat)
                    if os.path.isdir(cat_path):
                        exp_counts[cat] = exp_counts.get(cat, 0) + len(glob.glob(os.path.join(cat_path, "*.*")))
        results["archive (2) FER"] = {
            "task": "Facial Landmarks & Expressions",
            "categories": exp_counts
        }
        print(f"FER Archive (2): {exp_counts}")

    print("=== AUDIT COMPLETE ===")
    return results

if __name__ == "__main__":
    audit_datasets()
