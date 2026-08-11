import os
import sys
import glob
import torch
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfake_detection.inference import DeepfakeDetector
from explainability.gradcam import GradCAMExplainer
from explainability.edited_region_detector import EditedRegionDetector
from inference.attributes_predictor import FaceAttributePredictor

def test_pipeline():
    print("=== TESTING COMPLETE INTEGRATED INFERENCE PIPELINE ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    detector = DeepfakeDetector(image_model_path="models/best_model.pth" if os.path.exists("models/best_model.pth") else None)
    explainer = GradCAMExplainer(model=detector.image_model, target_layer=detector.image_model.efficientnet._conv_head)
    region_detector = EditedRegionDetector()
    attr_predictor = FaceAttributePredictor(model_path="models/best_attribute_model.pth" if os.path.exists("models/best_attribute_model.pth") else None)
    
    # Load sample image
    sample_images = glob.glob(r"c:\Users\Dell\Downloads\Datasets\archive (3)\**\*.jpg*", recursive=True)
    if not sample_images:
        sample_images = glob.glob(r"c:\Users\Dell\Downloads\Datasets\VIT_Dataset\**\*.png", recursive=True)
        
    if not sample_images:
        print("No sample test images found.")
        return

    test_path = sample_images[0]
    print(f"Testing on image: {test_path}")
    img = Image.open(test_path).convert('RGB')
    
    # 1. Authenticity Prediction
    df_res = detector.predict_image(img)
    print("\n1. AUTHENTICITY PREDICTION:")
    print(f"   Prediction: {df_res['prediction']}")
    print(f"   Confidence: {df_res['confidence']}% ({df_res['status']})")
    print(f"   Real Prob: {df_res['real_probability']}% | Fake Prob: {df_res['fake_probability']}%")
    
    # 2. Heatmap & Localization
    face_crop = df_res['face']
    face_tensor = detector.transform(face_crop).unsqueeze(0)
    vis, mask, raw = explainer.generate_heatmap(face_tensor, face_crop)
    reg_res = region_detector.detect_regions(face_crop, raw)
    
    print("\n2. MANIPULATION LOCALIZATION:")
    print(f"   Edited Detected: {reg_res['edited_detected']}")
    print(f"   Edited Area: {reg_res['edited_area_percentage']}%")
    print(f"   Suspicious Regions: {reg_res['suspicious_regions']}")
    print(f"   Explanations: {reg_res['explanations']}")
    
    # 3. Face Attributes Analysis
    attr_res = attr_predictor.predict_attributes(face_crop)
    print("\n3. FACE ATTRIBUTES ANALYSIS:")
    print(f"   Gender: {attr_res['gender']['prediction']} ({attr_res['gender']['confidence']}%)")
    print(f"   Face Shape: {attr_res['face_shape']['prediction']} ({attr_res['face_shape']['confidence']}%)")
    print(f"   Hair Texture: {attr_res['hair_texture']['prediction']} ({attr_res['hair_texture']['confidence']}%)")
    print(f"   Hair Color: {attr_res['hair_color']['prediction']} ({attr_res['hair_color']['confidence']}%)")
    print(f"   Skin Tone: {attr_res['skin_tone']['prediction']} ({attr_res['skin_tone']['confidence']}%)")
    
    print("\n=== COMPLETE PIPELINE VERIFIED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_pipeline()
