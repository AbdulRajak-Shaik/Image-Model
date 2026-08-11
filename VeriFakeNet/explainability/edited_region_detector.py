import cv2
import numpy as np
from PIL import Image

class EditedRegionDetector:
    def __init__(self):
        pass

    def detect_regions(self, face_image, raw_heatmap, ela_image=None):
        """
        Combines Grad-CAM activations, ELA anomalies, and edge inconsistencies
        to detect, localize, and highlight manipulated facial regions.
        
        Args:
            face_image (PIL.Image or np.ndarray): Extracted face image.
            raw_heatmap (np.ndarray): Grayscale heatmap [0, 255] (matching shape of face_image).
            ela_image (PIL.Image or np.ndarray): Error Level Analysis image (optional).
        """
        if not isinstance(face_image, np.ndarray):
            face_np = np.array(face_image.convert('RGB'))
        else:
            face_np = face_image.copy()
            
        h, w, _ = face_np.shape
        
        # 1. Edge detection using Sobel filter
        gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_magnitude = np.uint8(np.clip(edge_magnitude, 0, 255))
        
        # Resize raw heatmap to match face image
        if raw_heatmap.shape[:2] != (h, w):
            raw_heatmap_resized = cv2.resize(raw_heatmap, (w, h))
        else:
            raw_heatmap_resized = raw_heatmap.copy()
            
        # 2. Combine Grad-CAM heatmap and Edge inconsistencies
        gradcam_weight = 0.70
        edge_weight = 0.30
        
        combined_anomaly = (raw_heatmap_resized * gradcam_weight) + (edge_magnitude * edge_weight)
        
        # Integrate ELA compression artifacts if available
        if ela_image is not None:
            if not isinstance(ela_image, np.ndarray):
                ela_np = np.array(ela_image.convert('RGB'))
            else:
                ela_np = ela_image
            ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
            if ela_gray.shape[:2] != (h, w):
                ela_gray = cv2.resize(ela_gray, (w, h))
            combined_anomaly = (combined_anomaly * 0.80) + (ela_gray * 0.20)
            
        combined_anomaly = np.clip(combined_anomaly, 0, 255).astype(np.uint8)
        _, thresholded = cv2.threshold(combined_anomaly, 120, 255, cv2.THRESH_BINARY)
        
        # 3. Highlight manipulated regions on original image
        highlighted_image = face_np.copy()
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes and red anomaly highlights
        contour_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (h * w * 0.005): # Filter small noise
                contour_count += 1
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Draw glowing cyan/red bounding box
                cv2.rectangle(highlighted_image, (x, y), (x + bw, y + bh), (255, 0, 80), 2)
                # Overlay semi-transparent highlight
                sub_region = highlighted_image[y:y+bh, x:x+bw]
                red_tint = np.full_like(sub_region, (255, 30, 90), dtype=np.uint8)
                blended = cv2.addWeighted(sub_region, 0.65, red_tint, 0.35, 0)
                highlighted_image[y:y+bh, x:x+bw] = blended

        # 4. Calculate percentage of edited area
        edited_pixels = np.sum(thresholded > 0)
        total_pixels = h * w
        edited_percentage = (edited_pixels / total_pixels) * 100.0
        
        # 5. Facial zone mapping
        suspicious_regions = []
        explanation_bullets = []
        
        top_zone = thresholded[0:int(h*0.35), :]
        mid_zone = thresholded[int(h*0.35):int(h*0.70), :]
        bottom_zone = thresholded[int(h*0.70):, :]
        
        if np.sum(top_zone > 0) / (total_pixels * 0.35) > 0.04:
            suspicious_regions.append("Eye & Hairline Boundary")
            explanation_bullets.append("Pixel-level blending artifacts detected around the eyes and hairline.")
        if np.sum(mid_zone > 0) / (total_pixels * 0.35) > 0.04:
            suspicious_regions.append("Cheeks & Nose Region")
            explanation_bullets.append("Irregular skin texture and localized noise patterns detected along cheek/nose contours.")
        if np.sum(bottom_zone > 0) / (total_pixels * 0.30) > 0.04:
            suspicious_regions.append("Mouth & Jawline Boundary")
            explanation_bullets.append("Facial boundary inconsistencies and unnatural blending detected near the mouth/jaw area.")
            
        if not suspicious_regions and edited_percentage > 2.0:
            suspicious_regions.append("Face-Swap Seam Boundary")
            explanation_bullets.append("High-frequency compression edge seam detected across facial perimeter.")
        elif not suspicious_regions:
            explanation_bullets.append("No major localized manipulation anomalies detected across key facial zones.")
            
        return {
            "edited_detected": len(suspicious_regions) > 0,
            "edited_area_percentage": round(edited_percentage, 2),
            "suspicious_regions": suspicious_regions,
            "explanations": explanation_bullets,
            "highlighted_image": highlighted_image,
            "mask": thresholded
        }
