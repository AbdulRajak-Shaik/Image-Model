import torch
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from PIL import Image

class FaceDetector:
    def __init__(self, device=None, margin=20, image_size=300):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.mtcnn = MTCNN(
            keep_all=False, 
            device=self.device, 
            margin=margin, 
            image_size=image_size,
            post_process=False,
            thresholds=[0.4, 0.5, 0.5],
            min_face_size=15
        )
        # OpenCV Haar Cascade Fallback (if available in environment)
        try:
            self.cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') else None
            self.cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml') if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') else None
        except Exception:
            self.cascade_frontal = None
            self.cascade_profile = None

    def extract_face(self, image):
        """
        Extracts the largest face from a PIL Image or numpy array using MTCNN with fallback.
        Returns PIL image of the cropped face and the bounding box.
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # 1. Primary: MTCNN Detector
        try:
            boxes, probs = self.mtcnn.detect(image)
            if boxes is not None and len(boxes) > 0:
                box = [int(b) for b in boxes[0]]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3])
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    face_pil = image.crop((x1, y1, x2, y2)).resize((self.mtcnn.image_size, self.mtcnn.image_size), Image.BILINEAR)
                    return face_pil, [x1, y1, x2, y2]
        except Exception:
            pass

        # 2. Secondary: OpenCV Haar Cascade Fallback
        if self.cascade_frontal is not None:
            try:
                img_np = np.array(image)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
                faces = self.cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) == 0 and self.cascade_profile is not None:
                    faces = self.cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = faces[0]
                    margin_x, margin_y = int(fw * 0.2), int(fh * 0.2)
                    x1, y1 = max(0, fx - margin_x), max(0, fy - margin_y)
                    x2, y2 = min(image.width, fx + fw + margin_x), min(image.height, fy + fh + margin_y)
                    face_pil = image.crop((x1, y1, x2, y2)).resize((self.mtcnn.image_size, self.mtcnn.image_size), Image.BILINEAR)
                    return face_pil, [x1, y1, x2, y2]
            except Exception:
                pass

        # 3. Tertiary: Upper-Center Head Crop Heuristic (for portrait / waist-up shots)
        w, h = image.size
        if h > 100 and w > 100:
            x1, y1 = int(w * 0.15), int(h * 0.05)
            x2, y2 = int(w * 0.85), int(h * 0.65)
            face_pil = image.crop((x1, y1, x2, y2)).resize((self.mtcnn.image_size, self.mtcnn.image_size), Image.BILINEAR)
            return face_pil, [x1, y1, x2, y2]

        return None, None

    def extract_frames_faces(self, video_path, num_frames=20):
        """
        Extracts faces from a sequence of frames in a video.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return []
            
        # Select evenly spaced frames
        frame_idxs = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        faces = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            face_pil, box = self.extract_face(frame)
            if face_pil is not None:
                faces.append(face_pil)
                
        cap.release()
        return faces
