import torch
import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

class GradCAMExplainer:
    def __init__(self, model, target_layer, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def generate_heatmap(self, input_tensor, original_image):
        """
        Generates a GradCAM heatmap, overlay image, and a binary manipulation mask.
        Returns:
            visualization (np.ndarray): Heatmap overlaid on RGB image.
            mask (np.ndarray): Binary mask highlighting regions with activation > 0.5.
            raw_heatmap (np.ndarray): Grayscale heatmap in range [0, 255].
        """
        input_tensor = input_tensor.to(self.device)
        targets = [ClassifierOutputTarget(0)]
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        if not isinstance(original_image, np.ndarray):
            original_image = np.array(original_image)
            
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 4:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
            
        rgb_img = np.float32(original_image) / 255
        
        if rgb_img.shape[:2] != grayscale_cam.shape:
            grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
            
        # Overlay heatmap on image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Generate binary manipulation mask (threshold activation at 50%)
        binary_mask = (grayscale_cam > 0.5).astype(np.uint8) * 255
        
        raw_heatmap = (grayscale_cam * 255).astype(np.uint8)
        
        return visualization, binary_mask, raw_heatmap

    def generate_video_heatmap(self, video_path, face_detector, transform, output_path="outputs/heatmaps/heatmap_video.mp4", max_frames=60):
        """
        Processes video frames, runs face detection, extracts Grad-CAM heatmaps,
        and saves a compiled video.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_pil, box = face_detector.extract_face(frame_rgb)
            
            if face_pil is not None and box is not None:
                # Get Grad-CAM on cropped face
                face_tensor = transform(face_pil).unsqueeze(0).to(self.device)
                vis_face, _, _ = self.generate_heatmap(face_tensor, face_pil)
                
                # Resize vis_face back to bounding box sizes and overlay on the frame
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                vis_face_resized = cv2.resize(vis_face, (w, h))
                
                # Place back into the original frame frame_rgb
                frame_rgb[y1:y2, x1:x2] = vis_face_resized
                
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            frame_idx += 1
            
        cap.release()
        out.release()
        return output_path
