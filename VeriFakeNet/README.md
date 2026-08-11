# VeriFakeNet: A Unified Approach to Explainable Deepfake Detection and Media Authenticity Assessment

VeriFakeNet is a comprehensive, modular, and explainable deepfake detection system. It not only classifies images and videos as Real or Fake but also provides a multi-faceted authenticity assessment using deep learning and forensic techniques.

## Features
- **Deepfake Detection**: Uses EfficientNet-B3 and BiLSTM for robust feature extraction and sequence modeling.
- **Explainable AI (XAI)**: Employs Grad-CAM to visualize the facial regions most indicative of manipulation.
- **Error Level Analysis (ELA)**: Detects tampered regions through JPEG compression analysis.
- **Metadata Forensics**: Extracts and analyzes Exif data to identify discrepancies or editing software traces.
- **Perceptual Hash Verification**: Uses ImageHash for content integrity verification.
- **Trust Assessment Engine**: Aggregates all analyses into a comprehensive Trust Score (0-100).
- **Report Generation**: Automatically creates a detailed PDF report of the findings.
- **Interactive UI**: A professional Streamlit application for end-to-end media analysis.

## Setup

1. **Clone the repository** (if not already done).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install ExifTool**:
   You need to install ExifTool on your system for the Metadata Forensics module to work.
   - **Windows**: Download the Windows Executable from [ExifTool's official site](https://exiftool.org/), extract it, rename `exiftool(-k).exe` to `exiftool.exe`, and place it in a directory included in your system's PATH (or inside this project directory).
   - **Linux**: `sudo apt install libimage-exiftool-perl`
   - **macOS**: `brew install exiftool`

## Running the Application

```bash
streamlit run streamlit_app/app.py
```

## Structure
- `deepfake_detection/`: Core AI models.
- `explainability/`: Grad-CAM implementation.
- `metadata_analysis/`, `ela/`, `hash_verification/`: Forensic modules.
- `trust_engine/`: Logic for the final trust score.
- `streamlit_app/`: Streamlit frontend.
- `notebooks/`: Jupyter notebooks for model training and evaluation.
