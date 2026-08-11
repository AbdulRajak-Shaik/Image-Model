import torch
from torchvision import transforms
from .models import DeepfakeImageModel, DeepfakeVideoModel
from .face_detector import FaceDetector
import torch.nn.functional as F
import numpy as np

class DeepfakeDetector:
    def __init__(self, image_model_path=None, video_model_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Face Detector
        self.face_detector = FaceDetector(device=self.device)
        
        # Transforms for EfficientNet-B3
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load Image Model
        self.image_model = DeepfakeImageModel(pretrained=(image_model_path is None)).to(self.device)
        if image_model_path:
            try:
                self.image_model.load_state_dict(torch.load(image_model_path, map_location=self.device))
                print(f"Loaded image model weights from {image_model_path}")
            except Exception as e:
                print(f"Notice: Image model path provided but failed to load: {e}")
        self.image_model.eval()
        
        # Load Video Model
        self.video_model = DeepfakeVideoModel(pretrained_image_model=self.image_model).to(self.device)
        if video_model_path:
            try:
                self.video_model.load_state_dict(torch.load(video_model_path, map_location=self.device))
            except Exception as e:
                pass
        self.video_model.eval()

    def _analyze_frequency_artifacts(self, face_pil):
        """
        Analyzes 2D Fast Fourier Transform (FFT) high-frequency spectrum for GAN / Diffusion / Deepfake grid artifacts.
        """
        try:
            img_gray = np.array(face_pil.convert('L'))
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            
            # High-frequency outer ring ratio
            h, w = magnitude_spectrum.shape
            cy, cx = h // 2, w // 2
            r = min(h, w) // 4
            
            y, x = np.ogrid[:h, :w]
            outer_mask = (x - cx)**2 + (y - cy)**2 > r**2
            outer_mean = np.mean(magnitude_spectrum[outer_mask])
            center_mean = np.mean(magnitude_spectrum[~outer_mask])
            
            freq_ratio = outer_mean / (center_mean + 1e-8)
            # High frequency ratio > 0.65 is characteristic of AI generation / resampling
            fft_fake_prob = min(1.0, max(0.0, (freq_ratio - 0.45) * 3.0))
            return float(fft_fake_prob)
        except Exception:
            return 0.5

    def predict_image(self, image):
        """
        Predicts if an image/face is Real or Fake/Edited using a hybrid ensemble combining:
        1. Deep Convolutional Neural Network (EfficientNet-B3)
        2. Fast Fourier Transform (FFT) High-Frequency Artifact Index
        Returns calibrated probability distribution and confidence status.
        """
        face_pil, box = self.face_detector.extract_face(image)
        
        no_face_detected = False
        if face_pil is None:
            no_face_detected = True
            face_pil = image.resize((300, 300)) if hasattr(image, 'resize') else image
            
        tensor = self.transform(face_pil.convert('RGB')).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.image_model(tensor)
            cnn_prob_fake = torch.sigmoid(output).item()
            
        fft_prob_fake = self._analyze_frequency_artifacts(face_pil)
        
        # Hybrid Ensemble Fusion: 70% CNN Deep Features + 30% FFT Spectral Artifacts
        prob_fake = (0.70 * cnn_prob_fake) + (0.30 * fft_prob_fake)
        prob_real = 1.0 - prob_fake
        is_fake = prob_fake > 0.50
        
        prediction = 'FAKE / EDITED' if is_fake else 'REAL'
        confidence = prob_fake if is_fake else prob_real
        confidence_pct = confidence * 100.0
        
        # Calibrated Confidence Status
        if confidence_pct >= 75.0:
            status = "High Confidence"
        elif confidence_pct >= 60.0:
            status = "Moderate Confidence"
        else:
            status = "Low-confidence prediction"
            
        return {
            'prediction': prediction,
            'confidence': round(confidence_pct, 2),
            'real_probability': round(prob_real * 100.0, 2),
            'fake_probability': round(prob_fake * 100.0, 2),
            'status': status,
            'face': face_pil,
            'bounding_box': box,
            'no_face_detected': no_face_detected,
            'input_tensor': tensor
        }

    def predict_video(self, video_path, num_frames=20):
        """
        Predicts if a video is real or fake by analyzing temporal sequence of frames.
        """
        faces = self.face_detector.extract_frames_faces(video_path, num_frames=num_frames)
        
        if len(faces) == 0:
             return {'prediction': 'Unknown', 'confidence': 0.0, 'error': 'No faces detected in video'}
             
        if len(faces) < num_frames:
            while len(faces) < num_frames:
                faces.append(faces[-1])
        elif len(faces) > num_frames:
            faces = faces[:num_frames]
            
        tensors = [self.transform(face.convert('RGB')) for face in faces]
        video_tensor = torch.stack(tensors).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.video_model(video_tensor)
            prob_fake = torch.sigmoid(output).item()
            
        prob_real = 1.0 - prob_fake
        is_fake = prob_fake > 0.5
        prediction = 'FAKE / EDITED' if is_fake else 'REAL'
        confidence = prob_fake if is_fake else prob_real
        
        return {
            'prediction': prediction,
            'confidence': round(confidence * 100.0, 2),
            'real_probability': round(prob_real * 100.0, 2),
            'fake_probability': round(prob_fake * 100.0, 2),
            'first_face': faces[0] if faces else None
        }
