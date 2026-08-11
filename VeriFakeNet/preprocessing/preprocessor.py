import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
from deepfake_detection.face_detector import FaceDetector

class MediaPreprocessor:
    def __init__(self, target_size=(300, 300), cache_dir="datasets/cache", device=None):
        self.target_size = target_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_detector = FaceDetector(device=self.device, image_size=target_size[0])

        # Image augmentation pipeline
        self.train_transforms = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Normal validation/test transforms
        self.val_transforms = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path, is_training=True):
        """Loads, detects face, crops, and applies PyTorch transforms."""
        try:
            img = Image.open(image_path).convert('RGB')
            face_pil, _ = self.face_detector.extract_face(img)
            
            # Fallback if MTCNN fails to find a face
            if face_pil is None:
                face_pil = img.resize(self.target_size, Image.BILINEAR)

            if is_training:
                return self.train_transforms(face_pil)
            else:
                return self.val_transforms(face_pil)
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None

    def preprocess_video_and_cache(self, video_path, cache_name, num_frames=20):
        """
        Extracts frames, crops faces, and saves them as a numpy cache (.npy)
        to bypass MTCNN face detection overhead in subsequent epochs.
        """
        cache_file = self.cache_dir / f"{cache_name}.npy"
        
        if cache_file.exists():
            # Load from cache
            try:
                cached_faces = np.load(cache_file, allow_pickle=True)
                return [Image.fromarray(face) for face in cached_faces]
            except Exception as e:
                print(f"Failed to load cache {cache_file}: {e}. Processing video instead.")

        # Extract frames & run MTCNN
        faces = self.face_detector.extract_frames_faces(str(video_path), num_frames=num_frames)
        
        if not faces:
            return []

        # Convert PIL images to numpy arrays for serialization
        faces_np = np.array([np.array(face) for face in faces], dtype=object)
        
        # Save to cache
        try:
            np.save(cache_file, faces_np)
        except Exception as e:
            print(f"Could not save cache file: {e}")
            
        return faces
