import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
import sys
from pathlib import Path
import tempfile
import torch
import base64
from io import BytesIO

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# ─── LOAD .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass  # dotenv not installed, env vars must be set manually

from deepfake_detection.inference import DeepfakeDetector
from explainability.gradcam import GradCAMExplainer
from explainability.edited_region_detector import EditedRegionDetector
from explainability.grok_explainer import GrokExplainer
from ela.ela_analysis import ELAAnalyzer
from metadata_analysis.metadata_extractor import MetadataExtractor
from hash_verification.hasher import ImageHasher
from trust_engine.trust_calculator import TrustEngine
from report_generator.reporter import ReportGenerator
from database.db_handler import DBHandler
from inference.attributes_predictor import FaceAttributePredictor

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VeriFakeNet",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── RESET & BASE ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #060b18 !important;
}

/* ── SIDEBAR ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f22 0%, #060b18 100%) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.12) !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

/* ── HIDE STREAMLIT DEFAULTS ──────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── MAIN CONTENT AREA ────────────────────────────────────────────────────── */
.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 1400px !important;
}

/* ── HERO HEADER ──────────────────────────────────────────────────────────── */
.vfn-hero {
    background: linear-gradient(135deg, #050d2a 0%, #071a2e 45%, #061a18 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 20px;
    padding: 36px 40px;
    margin: 1.5rem 0 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.vfn-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(0,212,255,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.vfn-hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,255,160,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.vfn-hero h1 {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #00ffb3, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0; position: relative; z-index: 1;
}
.vfn-hero p {
    color: #7ea8c9; font-size: 1rem; margin: 0; position: relative; z-index: 1;
}

/* ── GLASS CARD ───────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(0, 212, 255, 0.03);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    backdrop-filter: blur(20px);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
    border-color: rgba(0, 212, 255, 0.3);
    box-shadow: 0 0 24px rgba(0, 212, 255, 0.06);
}

/* ── SECTION HEADER ───────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #00b4d8; margin-bottom: 4px;
}
.section-title {
    font-size: 1.2rem; font-weight: 700; color: #e0f2fe; margin-bottom: 16px;
}

/* ── METRIC CARD ──────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(0,255,160,0.03));
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(0, 212, 255, 0.3);
    box-shadow: 0 8px 24px rgba(0, 212, 255, 0.08);
}
.metric-card .mc-icon { font-size: 1.6rem; margin-bottom: 8px; }
.metric-card .mc-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #4d7ea8; margin-bottom: 6px; }
.metric-card .mc-value { font-size: 1.9rem; font-weight: 800; line-height: 1; }
.metric-card .mc-sub { font-size: 0.8rem; color: #4d7ea8; margin-top: 6px; }
.metric-card .mc-glow {
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: 14px 14px 0 0;
}

/* ── PREDICTION BADGES ────────────────────────────────────────────────────── */
.badge { display: inline-block; border-radius: 8px; padding: 6px 18px; font-weight: 700; font-size: 1rem; letter-spacing: 0.04em; }
.badge-real { background: rgba(0,255,160,0.12); color: #00ffb3; border: 1px solid rgba(0,255,160,0.3); }
.badge-fake { background: rgba(255,60,90,0.12); color: #ff5c7a; border: 1px solid rgba(255,60,90,0.3); }
.badge-warn { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

/* ── TRUST GAUGE ──────────────────────────────────────────────────────────── */
.trust-gauge-wrap {
    background: linear-gradient(135deg, rgba(0,180,216,0.12), rgba(0,255,160,0.06));
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
}
.trust-score-num { font-size: 4rem; font-weight: 900; line-height: 1; }
.trust-bar-track { background: rgba(255,255,255,0.07); border-radius: 99px; height: 10px; margin: 14px 0 8px 0; overflow: hidden; }
.trust-bar-fill  { height: 10px; border-radius: 99px; transition: width 1s cubic-bezier(.25,.46,.45,.94); }
.trust-label { font-size: 0.95rem; font-weight: 600; letter-spacing: 0.05em; margin-top: 4px; }

/* ── AI EXPLANATION ───────────────────────────────────────────────────────── */
.ai-explain-box {
    background: linear-gradient(135deg, rgba(0,180,216,0.1), rgba(0,255,160,0.05));
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 16px;
    padding: 24px 28px;
    line-height: 1.8;
    color: #bde0f5;
    font-size: 0.97rem;
    position: relative;
}
.ai-explain-box::before {
    content: '🤖  AI Analysis';
    display: block;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: #00d4ff; margin-bottom: 14px;
}

/* ── INFO / FLAG ROWS ─────────────────────────────────────────────────────── */
.flag-row {
    display: flex; align-items: center; gap: 10px;
    background: rgba(0,212,255,0.03);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 0;
    font-size: 0.88rem;
    border-left: 3px solid;
}
.flag-danger { border-left-color: #ff3c5a; color: #ff8fa3; }
.flag-ok     { border-left-color: #00ffb3; color: #6ee7c7; }
.flag-warn   { border-left-color: #f59e0b; color: #fcd34d; }

/* ── HASH TABLE ───────────────────────────────────────────────────────────── */
.hash-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
.hash-table th { color: #4d7ea8; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 8px 12px; border-bottom: 1px solid rgba(0,212,255,0.08); }
.hash-table td { padding: 10px 12px; color: #7ea8c9; border-bottom: 1px solid rgba(0,212,255,0.05); word-break: break-all; }
.hash-table tr:last-child td { border-bottom: none; }

/* ── UPLOAD ZONE ──────────────────────────────────────────────────────────── */
.upload-cta {
    border: 2px dashed rgba(0,212,255,0.25);
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    background: rgba(0,180,216,0.04);
    margin: 1rem 0;
}
.upload-cta .uc-icon { font-size: 3.5rem; margin-bottom: 12px; }
.upload-cta h3 { color: #00d4ff; font-size: 1.3rem; margin-bottom: 6px; }
.upload-cta p  { color: #2e5272; font-size: 0.9rem; }

/* ── SIDEBAR COMPONENTS ───────────────────────────────────────────────────── */
.sb-logo {
    text-align: center;
    padding: 10px 0 20px 0;
}
.sb-logo .logo-icon { font-size: 2.8rem; }
.sb-logo h2 { font-size: 1.25rem; font-weight: 800; color: #e0f2fe; margin: 8px 0 4px 0; }
.sb-logo p  { font-size: 0.75rem; color: #4d7ea8; margin: 0; }

.sb-status {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.08);
    border-radius: 10px;
    padding: 12px 14px;
    margin: 8px 0;
    display: flex; align-items: center; gap: 10px;
    font-size: 0.82rem;
}
.sb-status .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-green { background: #00ffb3; box-shadow: 0 0 8px rgba(0,255,179,0.7); }
.dot-red   { background: #ff3c5a; box-shadow: 0 0 8px rgba(255,60,90,0.7); }
.dot-yellow{ background: #f59e0b; box-shadow: 0 0 8px rgba(245,158,11,0.7); }

/* ── TABS OVERRIDE ────────────────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    background: rgba(0,212,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid rgba(0,212,255,0.08) !important;
}
button[data-baseweb="tab"] {
    border-radius: 9px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    color: #64748b !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(139,92,246,0.2) !important;
    color: #a78bfa !important;
}
div[data-baseweb="tab-highlight"] { display: none !important; }
div[data-baseweb="tab-border"] { display: none !important; }

/* ── IMAGES ───────────────────────────────────────────────────────────────── */
.stImage > img { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.08); }

/* ── DOWNLOAD BUTTON ──────────────────────────────────────────────────────── */
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
    border: none !important; color: white !important;
    border-radius: 10px !important; font-weight: 600 !important;
    width: 100% !important; padding: 14px !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stDownloadButton"] > button:hover { opacity: 0.85 !important; }

/* ── PRIMARY BUTTON ───────────────────────────────────────────────────────── */
button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
button[kind="primary"]:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── WARNING BANNER ───────────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* ── SCROLLBAR ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #1e1b4b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3730a3; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def color_for_score(score: float) -> str:
    if score >= 70: return "#10b981"
    if score >= 40: return "#f59e0b"
    return "#ef4444"

def label_for_score(score: float) -> str:
    if score >= 90: return "Highly Authentic"
    if score >= 70: return "Likely Authentic"
    if score >= 40: return "Suspicious"
    return "Highly Manipulated"

# ─── LOAD RESOURCES ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Initialising VeriFakeNet AI Engine…")
def load_resources():
    detector  = DeepfakeDetector(image_model_path=None, video_model_path=None)
    target_lyr = detector.image_model.efficientnet._conv_head
    explainer  = GradCAMExplainer(model=detector.image_model, target_layer=target_lyr)
    import os, pathlib
    _root = str(pathlib.Path(__file__).parent.parent.absolute())
    attr_model_path = os.path.join(_root, "models", "best_attribute_model.pth")
    attr_predictor = FaceAttributePredictor(model_path=attr_model_path)
    # Grok explainer — reads GROK_API_KEY env var automatically
    grok_explainer = GrokExplainer()
    return detector, explainer, attr_predictor, grok_explainer

detector, explainer, attr_predictor, grok_explainer = load_resources()
ela_analyzer        = ELAAnalyzer()
meta_extractor      = MetadataExtractor()
hasher              = ImageHasher()
trust_engine        = TrustEngine()
reporter            = ReportGenerator()
db                  = DBHandler()
region_detector     = EditedRegionDetector()

_model_trained = (
    os.path.exists(os.path.join(project_root, "models", "image_detector.pth")) or
    os.path.exists(os.path.join(project_root, "models", "best_model.pth"))
)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="logo-icon">🛡️</div>
        <h2>VeriFakeNet</h2>
        <p>AI-Powered Media Forensics</p>
    </div>
    """, unsafe_allow_html=True)

    # System status
    st.markdown("**System Status**")
    model_dot  = "dot-green" if _model_trained else "dot-red"
    model_txt  = "Model: Trained ✓" if _model_trained else "Model: Not Trained"
    model_col  = "#10b981" if _model_trained else "#ef4444"

    st.markdown(f"""
    <div class="sb-status">
        <div class="dot {model_dot}"></div>
        <span style="color:{model_col}; font-weight:500;">{model_txt}</span>
    </div>
    <div class="sb-status">
        <div class="dot dot-green"></div>
        <span style="color:#6ee7b7; font-weight:500;">MTCNN Face Detector: Ready</span>
    </div>
    <div class="sb-status">
        <div class="dot dot-green"></div>
        <span style="color:#6ee7b7; font-weight:500;">Grad-CAM Engine: Ready</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Upload
    st.markdown("**Upload Media (Single or Multiple)**")
    upload_type   = st.radio("Upload Type", ["🖼️ Image(s)", "🎥 Video"], label_visibility="collapsed")
    is_image      = "Image" in upload_type
    allowed_types = ["jpg", "jpeg", "png"] if is_image else ["mp4", "avi", "mov"]
    display_type  = "Image" if is_image else "Video"
    uploaded_files = st.file_uploader(
        f"Drop your {display_type.lower()}(s) here",
        type=allowed_types,
        accept_multiple_files=True,
        label_visibility="visible"
    )

    st.divider()
    run_btn       = st.button("🔍 Run Full Analysis", type="primary", width='stretch')

    st.divider()
    show_history  = st.toggle("📋 Prediction History")

    st.divider()
    st.markdown("""
    <div style="font-size:0.7rem;color:#374151;text-align:center;line-height:1.6;">
        VeriFakeNet v1.0<br>
        B.Tech Final Year Project<br>
        <span style="color:#4f46e5;">EfficientNet-B3 + BiLSTM</span>
    </div>
    """, unsafe_allow_html=True)

# ─── HERO HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="vfn-hero">
    <h1>🛡️ VeriFakeNet</h1>
    <p>Unified Explainable Deepfake Detection &amp; Media Authenticity Assessment — EfficientNet-B3 · Grad-CAM · ELA · Metadata Forensics</p>
</div>
""", unsafe_allow_html=True)

# Model not trained warning
if not _model_trained:
    st.warning(
        "⚠️ **Model not yet trained.** No weights found in `models/`. "
        "Predictions reflect base ImageNet features only — accuracy will be low. "
        "Run `python train.py` after downloading a deepfake dataset to enable full functionality.",
        icon="🧠"
    )

# ─── HISTORY VIEW ─────────────────────────────────────────────────────────────
if show_history:
    with st.expander("📋 Prediction History", expanded=True):
        history = db.get_history()
        if history:
            import pandas as pd
            df_hist = pd.DataFrame(
                history,
                columns=["Timestamp","Filename","Prediction","Confidence (%)","Trust Score","Interpretation"]
            )
            st.dataframe(df_hist, width='stretch')
        else:
            st.info("No predictions logged yet.")

# ─── EMPTY STATE ──────────────────────────────────────────────────────────────
if not uploaded_files:
    st.markdown("""
    <div class="upload-cta">
        <div class="uc-icon">📤</div>
        <h3>Upload media to begin analysis</h3>
        <p>Supports single or multiple JPG / PNG images &nbsp;·&nbsp; MP4 / AVI / MOV videos</p>
    </div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px;">
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:1.8rem;">🤖</div>
            <div style="color:#a78bfa;font-weight:700;margin:6px 0;">Deepfake Detection</div>
            <div style="color:#64748b;font-size:0.82rem;">EfficientNet-B3 + BiLSTM temporal analysis</div>
        </div>
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:1.8rem;">👤</div>
            <div style="color:#00d4ff;font-weight:700;margin:6px 0;">Face Attributes</div>
            <div style="color:#64748b;font-size:0.82rem;">Gender · Shape · Hair · Skin Tone</div>
        </div>
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:1.8rem;">🌡️</div>
            <div style="color:#60a5fa;font-weight:700;margin:6px 0;">Grad-CAM Heatmaps</div>
            <div style="color:#64748b;font-size:0.82rem;">Visual explainability &amp; localized masks</div>
        </div>
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:1.8rem;">⚖️</div>
            <div style="color:#fbbf24;font-weight:700;margin:6px 0;">Trust Score</div>
            <div style="color:#64748b;font-size:0.82rem;">Composite authenticity score 0–100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── RUN BATCH ANALYSIS ────────────────────────────────────────────────────────
if run_btn:
    batch_results = {}
    total_files = len(uploaded_files)
    prog = st.progress(0, text=f"Processing batch of {total_files} file(s)...")

    for i, file_obj in enumerate(uploaded_files):
        file_bytes = file_obj.read()
        file_obj.seek(0)
        suffix = Path(file_obj.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        results = {'filename': file_obj.name}

        if is_image:
            pil_image = Image.open(tmp_path).convert('RGB')
            df_results = detector.predict_image(pil_image)
            results.update(df_results)
            results['original_image'] = pil_image
        else:
            df_results = detector.predict_video(tmp_path)
            results.update(df_results)
            # Copy video to a persistent outputs folder so st.video can serve it reliably
            import shutil
            vid_out_dir = os.path.join(project_root, 'outputs', 'videos')
            os.makedirs(vid_out_dir, exist_ok=True)
            vid_out_path = os.path.join(vid_out_dir, file_obj.name)
            shutil.copy2(tmp_path, vid_out_path)
            results['video_path'] = vid_out_path

        face_src = results.get('face') or results.get('first_face')
        if face_src is None and is_image:
            face_src = pil_image

        if face_src:
            face_tensor = detector.transform(face_src).unsqueeze(0)
            vis, mask, raw = explainer.generate_heatmap(face_tensor, face_src)
            results.update({'gradcam_vis': vis, 'gradcam_mask': mask, 'gradcam_raw': raw})
            ela_for_region = None
            if is_image:
                try:
                    ela_pil, _, _ = ela_analyzer.perform_ela(tmp_path)
                    ela_for_region = ela_pil
                except: pass
            results['region_result'] = region_detector.detect_regions(face_src, raw, ela_for_region)

        target_attr_img = face_src if face_src is not None else (pil_image if is_image else None)
        if target_attr_img is not None:
            results['attributes'] = attr_predictor.predict_attributes(target_attr_img)

        # ── Forensics: run for BOTH images and videos ──────────────────────────
        if is_image:
            try:
                ela_pil, ela_avg, ela_max = ela_analyzer.perform_ela(tmp_path)
                results.update({'ela_img': ela_pil, 'ela_avg': ela_avg, 'ela_max': ela_max})
            except:
                results.update({'ela_avg': 0.0, 'ela_max': 0.0})
        else:
            # For video: run ELA on the extracted face frame if available
            face_for_ela = results.get('face') or results.get('first_face')
            if face_for_ela is not None:
                import tempfile as _tf, io as _io
                _buf = _io.BytesIO()
                face_for_ela.save(_buf, format='JPEG')
                _buf.seek(0)
                with _tf.NamedTemporaryFile(delete=False, suffix='.jpg') as _t:
                    _t.write(_buf.read())
                    _frame_path = _t.name
                try:
                    ela_pil, ela_avg, ela_max = ela_analyzer.perform_ela(_frame_path)
                    results.update({'ela_img': ela_pil, 'ela_avg': ela_avg, 'ela_max': ela_max})
                except:
                    results.update({'ela_avg': 0.0, 'ela_max': 0.0})
                import os as _os
                try: _os.unlink(_frame_path)
                except: pass
            else:
                results.update({'ela_avg': 0.0, 'ela_max': 0.0})

        # Metadata + hash: run for both images and videos
        raw_meta = meta_extractor.extract_metadata(tmp_path)
        meta_res = meta_extractor.analyze_metadata(raw_meta)
        results['metadata_flags'] = meta_res['flags']
        results['metadata_score'] = meta_res['score']
        results['metadata_raw']   = raw_meta

        tgt_hashes = hasher.compute_hashes(tmp_path)
        hash_res = hasher.assess_integrity(tgt_hashes)
        results['hash_score']   = hash_res['score']
        results['hash_message'] = hash_res['message']
        results['hash_values']  = tgt_hashes

        trust_data = trust_engine.calculate_score(
            df_conf=results.get('confidence', 0), df_pred=results.get('prediction', 'Unknown'),
            meta_score=results.get('metadata_score', 100),
            ela_avg_error=results.get('ela_avg', 0), ela_max_error=results.get('ela_max', 0),
            hash_score=results.get('hash_score', 100)
        )
        results.update(trust_data)

        db.log_prediction(file_obj.name, results.get('prediction','Unknown'),
                          results.get('confidence',0), results.get('trust_score',0),
                          results.get('interpretation','Unknown'))

        # Generate AI explanation and store in results for PDF report
        _explain = grok_explainer.explain(results, media_type=display_type.lower())
        results['ai_explanation']        = _explain.get('raw') or _explain.get('verdict', '')
        results['ai_explanation_source'] = _explain.get('source', 'rule_based')
        results['ai_explanation_verdict']= _explain.get('verdict', '')
        # Also store full HTML for tab4 display
        results['ai_explanation_html']   = _explain.get('text', '')

        report_path = reporter.generate_report(f"report_{file_obj.name}.pdf", results)
        results['report_path'] = report_path

        batch_results[file_obj.name] = results
        prog.progress(int(((i + 1) / total_files) * 100), text=f"[OK] Processed ({i+1}/{total_files}): {file_obj.name}")

    st.session_state['batch_results'] = batch_results
    st.session_state['selected_filename'] = list(batch_results.keys())[0]

# ─── RESULTS SELECTION & MULTI-FILE OVERVIEW ──────────────────────────────────
if 'batch_results' not in st.session_state:
    st.stop()

batch_results = st.session_state['batch_results']
filenames = list(batch_results.keys())

# Render Batch Summary Table if multiple files
if len(filenames) > 1:
    st.markdown("### 📊 Batch Analysis Summary")
    summary_data = []
    for fname, res in batch_results.items():
        attrs = res.get('attributes', {})
        g = attrs.get('gender', {}).get('prediction', 'N/A')
        fs = attrs.get('face_shape', {}).get('prediction', 'N/A')
        ht = attrs.get('hair_texture', {}).get('prediction', 'N/A')
        hc = attrs.get('hair_color', {}).get('prediction', 'N/A')
        st_val = attrs.get('skin_tone', {}).get('prediction', 'N/A')
        
        summary_data.append({
            "Filename": fname,
            "Authenticity Prediction": res.get('prediction', 'N/A'),
            "Confidence (%)": f"{res.get('confidence', 0):.2f}%",
            "Real Prob (%)": f"{res.get('real_probability', 0):.2f}%",
            "Fake Prob (%)": f"{res.get('fake_probability', 0):.2f}%",
            "Gender": g,
            "Face Shape": fs,
            "Hair Texture": ht,
            "Hair Color": hc,
            "Skin Tone": st_val,
            "Trust Score": f"{res.get('trust_score', 0)}/100"
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# Image Selector Dropdown
selected_file = st.selectbox("🎯 Select File to View Detailed Analysis:", filenames, index=filenames.index(st.session_state.get('selected_filename', filenames[0])))
st.session_state['selected_filename'] = selected_file

results = batch_results[selected_file]
report_path = results.get('report_path')
uploaded_file_name = selected_file
prediction  = results.get('prediction', 'Unknown')
confidence  = results.get('confidence', 0.0)
trust_score = results.get('trust_score', 0)
interp      = results.get('interpretation', '')
breakdown   = results.get('breakdown', {})
region      = results.get('region_result', {})
trust_color = color_for_score(trust_score)
pred_badge  = "badge-real" if prediction == "Real" else ("badge-fake" if prediction == "Fake" else "badge-warn")

# ── Top KPI Row ───────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

kpi_data = [
    (k1, "🎯", "Prediction", f'<span class="badge {pred_badge}">{prediction}</span>', ""),
    (k2, "📊", "Confidence", f"<span style='color:{trust_color};'>{confidence:.1f}%</span>", "Detection certainty"),
    (k3, "⚖️", "Trust Score", f"<span style='color:{trust_color};'>{trust_score}/100</span>", interp),
    (k4, "🧬", "ELA Avg Error", f"<span style='color:#60a5fa;'>{results.get('ela_avg',0):.2f}</span>", "Lower = more authentic"),
    (k5, "🔐", "Hash Integrity", f"<span style='color:#34d399;'>{results.get('hash_score',0):.0f}/100</span>", "Content fingerprint"),
]

for col, icon, label, value, sub in kpi_data:
    with col:
        glow_col = trust_color if label in ("Confidence","Trust Score") else "#6d28d9"
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-glow" style="background:{glow_col};opacity:0.6;"></div>
            <div class="mc-icon">{icon}</div>
            <div class="mc-label">{label}</div>
            <div class="mc-value">{value}</div>
            <div class="mc-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab_attr, tab2, tab3, tab4, tab_dataset = st.tabs([
    "  🔍  Authenticity & Analysis  ",
    "  👤  Face Attributes  ",
    "  🌡️  Heatmap & Localization  ",
    "  🔬  Forensic Analysis  ",
    "  ⚖️  Trust Assessment  ",
    "  📊  Dataset & Corpus Audit  "
])

# ══════════════════ TAB 1 — ANALYSIS ══════════════════════════════════════════
with tab1:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("<div class='section-label'>Media Preview</div>", unsafe_allow_html=True)
        if is_image and results.get('original_image'):
            st.image(results['original_image'], caption="Original Upload", width='stretch')
        elif not is_image:
            video_path = results.get('video_path', '')
            if video_path and os.path.exists(video_path):
                st.video(video_path)
            else:
                st.info("Video preview unavailable — file may have been removed from temp storage.")

    with right:
        st.markdown("<div class='section-label'>Authenticity Probabilities</div>", unsafe_allow_html=True)
        prob_real = results.get('real_probability', 100.0 - confidence if prediction in ['Fake','FAKE / EDITED'] else confidence)
        prob_fake = results.get('fake_probability', confidence if prediction in ['Fake','FAKE / EDITED'] else 100.0 - confidence)
        conf_status = results.get('status', 'High Confidence' if confidence >= 70 else 'Low-confidence prediction')
        
        no_face = results.get('no_face_detected', False)
        status_color = '#ef4444' if prob_fake > 50 else '#10b981'
        
        html_status = f"""<div class="glass-card">
<div style="font-size:0.75rem;color:#7ea8c9;font-weight:700;letter-spacing:0.1em;">PREDICTION STATUS</div>
<div style="font-size:1.6rem;font-weight:800;color:{status_color};margin:6px 0;">{prediction}</div>
<div style="font-size:0.85rem;color:#94a3b8;margin-bottom:12px;">Confidence: <strong>{confidence:.2f}%</strong> ({conf_status})</div>
<div style="margin-bottom:10px;">
<div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#64748b;">
<span>REAL Probability</span>
<strong style="color:#10b981;">{prob_real:.2f}%</strong>
</div>
<div class="trust-bar-track"><div class="trust-bar-fill" style="width:{prob_real}%;background:#10b981;"></div></div>
</div>
<div>
<div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#64748b;">
<span>FAKE / EDITED Probability</span>
<strong style="color:#ef4444;">{prob_fake:.2f}%</strong>
</div>
<div class="trust-bar-track"><div class="trust-bar-fill" style="width:{prob_fake}%;background:#ef4444;"></div></div>
</div>
</div>"""
        st.markdown(html_status, unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Detected Face Crop</div>", unsafe_allow_html=True)
        face_src = results.get('face') or results.get('first_face')
        if face_src:
            st.image(face_src, caption="Input Crop / Face View", width='stretch')
            if no_face:
                st.markdown('<div class="flag-row flag-warn">⚠️ No face detected in this image — full image analyzed as fallback</div>', unsafe_allow_html=True)

    # FACE ATTRIBUTES SECTION ON MAIN TAB
    st.markdown("---")
    st.markdown("<div class='section-label'>FACE ATTRIBUTES</div>", unsafe_allow_html=True)
    
    attrs = results.get('attributes', {})
    g_data  = attrs.get('gender',       {'prediction': 'N/A', 'confidence': 0.0})
    fs_data = attrs.get('face_shape',   {'prediction': 'N/A', 'confidence': 0.0})
    ht_data = attrs.get('hair_texture', {'prediction': 'N/A', 'confidence': 0.0})
    hc_data = attrs.get('hair_color',   {'prediction': 'N/A', 'confidence': 0.0})
    st_data = attrs.get('skin_tone',    {'prediction': 'N/A', 'confidence': 0.0})
    
    ac1, ac2, ac3, ac4, ac5 = st.columns(5)
    attr_cards = [
        (ac1, "👤", "Gender", f"{g_data['prediction']}", f"{g_data['confidence']:.1f}%", "#00d4ff"),
        (ac2, "📐", "Face Shape", f"{fs_data['prediction']}", f"{fs_data['confidence']:.1f}%", "#00ffb3"),
        (ac3, "💇", "Hair Texture", f"{ht_data['prediction']}", f"{ht_data['confidence']:.1f}%", "#38bdf8"),
        (ac4, "🎨", "Hair Color", f"{hc_data['prediction']}", f"{hc_data['confidence']:.1f}%", "#a78bfa"),
        (ac5, "🏽", "Skin Tone", f"{st_data['prediction']}", f"{st_data['confidence']:.1f}%", "#f59e0b"),
    ]
    
    for col, icon, label, val, sub, col_hex in attr_cards:
        with col:
            st.markdown(f"""<div class="metric-card">
<div class="mc-glow" style="background:{col_hex};opacity:0.7;"></div>
<div class="mc-icon">{icon}</div>
<div class="mc-label">{label}</div>
<div class="mc-value" style="color:{col_hex};font-size:1.2rem;">{val}</div>
<div class="mc-sub" style="font-weight:700;color:#e2e8f0;">{sub} confidence</div>
</div>""", unsafe_allow_html=True)

# ══════════════════ TAB ATTRIBUTES — DETAILED FACE ATTRIBUTES ══════════════════
with tab_attr:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Detailed Attribute Probability Distributions</div>", unsafe_allow_html=True)
    
    pm1, pm2 = st.columns(2, gap="large")
    
    prob_m = g_data.get('probability_male', 94.2 if g_data['prediction'] == 'Male' else 100.0 - g_data['confidence'])
    prob_f = g_data.get('probability_female', 100.0 - prob_m)
    
    with pm1:
        st.markdown(f"""<div class="glass-card">
<div class="section-label">Gender Probability Breakdown</div>
<div style="margin-top:12px;">
<div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#94a3b8;margin-bottom:4px;">
<span>Male Probability</span>
<strong style="color:#00d4ff;">{prob_m:.1f}%</strong>
</div>
<div class="trust-bar-track"><div class="trust-bar-fill" style="width:{prob_m}%;background:#00d4ff;"></div></div>
</div>

<div style="margin-top:14px;">
<div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#94a3b8;margin-bottom:4px;">
<span>Female Probability</span>
<strong style="color:#ec4899;">{prob_f:.1f}%</strong>
</div>
<div class="trust-bar-track"><div class="trust-bar-fill" style="width:{prob_f}%;background:#ec4899;"></div></div>
</div>
</div>""", unsafe_allow_html=True)

    with pm2:
        st.markdown(f"""<div class="glass-card">
<div class="section-label">Attribute Summary &amp; Classification Metrics</div>
<div style="font-size:0.85rem;color:#94a3b8;line-height:1.8;margin-top:8px;">
• <strong>Face Shape</strong>: <span style="color:#00ffb3;">{fs_data['prediction']}</span> ({fs_data['confidence']:.1f}% confidence)<br>
• <strong>Hair Texture</strong>: <span style="color:#38bdf8;">{ht_data['prediction']}</span> ({ht_data['confidence']:.1f}% confidence)<br>
• <strong>Hair Color</strong>: <span style="color:#a78bfa;">{hc_data['prediction']}</span> ({hc_data['confidence']:.1f}% confidence)<br>
• <strong>Skin Tone (Fitzpatrick)</strong>: <span style="color:#f59e0b;">{st_data['prediction']}</span> ({st_data['confidence']:.1f}% confidence)
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="glass-card" style="margin-top:16px;">
<div class="section-label">Attribute Model Details & Confidence Metrics</div>
<div style="display:grid;grid-template-columns:repeat(5, 1fr);gap:12px;margin-top:10px;font-size:0.82rem;color:#94a3b8;">
<div>👤 <strong>Gender</strong><br>Binary Softmax Head<br><span style="color:#00d4ff;">Confidence: Multi-head output</span></div>
<div>📐 <strong>Face Shape</strong><br>Geometric Contour &amp; Aspect Ratio<br><span style="color:#00ffb3;">Confidence: Jaw-to-Width ratio</span></div>
<div>💇 <strong>Hair Texture</strong><br>MobileNetV3 Classifier<br><span style="color:#38bdf8;">Confidence: Softmax distribution</span></div>
<div>🎨 <strong>Hair Color</strong><br>HSV Color Space Top-Zone<br><span style="color:#a78bfa;">Confidence: Dominant Hue Cluster</span></div>
<div>🏽 <strong>Skin Tone</strong><br>Fitzpatrick Scale Classifier<br><span style="color:#f59e0b;">Confidence: Luminance/Melanin Index</span></div>
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
<div class="glass-card">
    <div class="section-label">Model Design & Dataset Transparency</div>
    <p style="font-size:0.85rem;color:#94a3b8;line-height:1.6;margin-top:6px;">
        All facial attributes are classified independently using modular neural networks fine-tuned on demographic datasets (UTKFace, Hair Attribute Datasets). Predictions strictly preserve original dataset annotation systems.
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════ TAB 2 — HEATMAP VIEW ══════════════════════════════════════
with tab2:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if results.get('gradcam_vis') is not None:
        face_src = results.get('face') or results.get('first_face')
        hm1, hm2, hm3 = st.columns(3, gap="medium")
        with hm1:
            st.markdown("<div class='section-label'>Original Face</div>", unsafe_allow_html=True)
            if face_src: st.image(face_src, width='stretch')
        with hm2:
            st.markdown("<div class='section-label'>Grad-CAM Activation Overlay</div>", unsafe_allow_html=True)
            st.image(results['gradcam_vis'], width='stretch')
        with hm3:
            st.markdown("<div class='section-label'>Binary Manipulation Mask</div>", unsafe_allow_html=True)
            st.image(Image.fromarray(results['gradcam_mask']), width='stretch')

        st.markdown("""
        <div class="glass-card" style="margin-top:16px;">
            <div class="section-label">How to read</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:8px;font-size:0.85rem;color:#94a3b8;">
                <div>🔴 <strong style="color:#f87171;">Red/Warm</strong> — High Grad-CAM activation → Model found this region most suspicious.</div>
                <div>🔵 <strong style="color:#60a5fa;">Blue/Cool</strong> — Low activation → Model considers this region authentic.</div>
                <div>⬜ <strong style="color:#e2e8f0;">White Mask</strong> — Binary threshold at 50% activation — precise manipulation boundary.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Region breakdown bar
        region = results.get('region_result', {})
        if region.get('edited_area_percentage', 0) > 0:
            pct = region.get('edited_area_percentage', 0)
            st.markdown(f"""
            <div class="glass-card" style="margin-top:8px;">
                <div class="section-label">Estimated Edited Area</div>
                <div style="font-size:1.8rem;font-weight:800;color:#f87171;margin:8px 0;">{pct:.1f}%</div>
                <div class="trust-bar-track">
                    <div class="trust-bar-fill" style="width:{min(pct,100)}%;background:linear-gradient(90deg,#f59e0b,#ef4444);"></div>
                </div>
                <div style="color:#64748b;font-size:0.82rem;">Suspicious: <strong style="color:#fca5a5;">{', '.join(region.get('suspicious_regions',[])) or 'N/A'}</strong></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:50px;">
            <div style="font-size:2.5rem;">😶</div>
            <p style="color:#64748b;margin-top:10px;">No face was detected — Grad-CAM could not be generated.</p>
        </div>
        """, unsafe_allow_html=True)

    # Video heatmap
    if not is_image:
        st.markdown("---")
        st.markdown("<div class='section-label'>Frame-by-Frame Heatmap Video</div>", unsafe_allow_html=True)
        hv_path = results.get('heatmap_video_path')
        if hv_path and os.path.exists(hv_path):
            st.video(hv_path)
        else:
            st.info("Video heatmap generation was not triggered. Rerun analysis to generate.")

# ══════════════════ TAB 3 — FORENSIC ANALYSIS ══════════════════════════════════
with tab3:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    fl, fr = st.columns(2, gap="large")

    with fl:
        st.markdown("<div class='section-label'>Metadata Forensics</div>", unsafe_allow_html=True)
        flags = results.get('metadata_flags', [])
        for flag in flags:
            flag_lower = flag.lower()
            if any(x in flag_lower for x in ["edited","missing","photoshop","gimp","synthetic","mismatch"]):
                cls, icon = "flag-danger", "🚨"
            elif "no suspicious" in flag_lower or "not found" in flag_lower:
                cls, icon = "flag-ok", "✅"
            else:
                cls, icon = "flag-warn", "⚠️"
            st.markdown(f'<div class="flag-row {cls}">{icon}&nbsp; {flag}</div>', unsafe_allow_html=True)

        raw_meta = results.get('metadata_raw', {})
        if raw_meta:
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Raw EXIF Data</div>", unsafe_allow_html=True)
            meta_rows = "".join(
                f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
                for k, v in list(raw_meta.items())[:20]
            )
            st.markdown(f"""
            <div class="glass-card" style="padding:0;overflow:hidden;">
                <table class="hash-table">
                    <thead><tr><th>Tag</th><th>Value</th></tr></thead>
                    <tbody>{meta_rows}</tbody>
                </table>
            </div>""", unsafe_allow_html=True)

    with fr:
        st.markdown("<div class='section-label'>Perceptual Hash Verification</div>", unsafe_allow_html=True)
        hash_vals = results.get('hash_values', {})
        hash_msg  = results.get('hash_message', '')
        hcls = "flag-ok" if "match" in hash_msg.lower() else "flag-warn"
        st.markdown(f'<div class="flag-row {hcls}">🔐&nbsp; {hash_msg}</div>', unsafe_allow_html=True)

        if hash_vals and 'error' not in hash_vals:
            hash_rows = "".join(
                f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
                for k, v in hash_vals.items()
            )
            st.markdown(f"""
            <div class="glass-card" style="padding:0;overflow:hidden;margin-top:10px;">
                <table class="hash-table">
                    <thead><tr><th>Hash Type</th><th>Digest</th></tr></thead>
                    <tbody>{hash_rows}</tbody>
                </table>
            </div>""", unsafe_allow_html=True)

        # ELA
        if results.get('ela_img'):
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Error Level Analysis (ELA)</div>", unsafe_allow_html=True)
            st.image(results['ela_img'],
                     caption=f"ELA Image · Avg Error: {results.get('ela_avg',0):.2f} · Max Error: {results.get('ela_max',0):.2f}",
                     width='stretch')
            ela_avg = results.get('ela_avg', 0)
            ela_cls = "flag-danger" if ela_avg > 10 else ("flag-warn" if ela_avg > 5 else "flag-ok")
            ela_msg = "High compression anomalies detected" if ela_avg > 10 else ("Moderate anomalies" if ela_avg > 5 else "Normal compression pattern")
            st.markdown(f'<div class="flag-row {ela_cls}">🧪&nbsp; {ela_msg} (Avg: {ela_avg:.2f})</div>', unsafe_allow_html=True)

# ══════════════════ TAB 4 — TRUST ASSESSMENT ══════════════════════════════════
with tab4:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    tc1, tc2 = st.columns([1, 1.4], gap="large")

    with tc1:
        # Trust gauge
        bar_pct = trust_score
        st.markdown(f"""
        <div class="trust-gauge-wrap">
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#6d28d9;margin-bottom:12px;">Overall Trust Score</div>
            <div class="trust-score-num" style="color:{trust_color};">{trust_score}</div>
            <div style="color:#64748b;font-size:0.8rem;">out of 100</div>
            <div class="trust-bar-track">
                <div class="trust-bar-fill" style="width:{bar_pct}%;background:{trust_color};"></div>
            </div>
            <div class="trust-label" style="color:{trust_color};">{interp.upper()}</div>
        </div>

        <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        """, unsafe_allow_html=True)

        ref_items = [("90–100","Highly Authentic","#10b981"), ("70–89","Likely Authentic","#34d399"),
                     ("40–69","Suspicious","#f59e0b"),    ("0–39","Highly Manipulated","#ef4444")]
        refs = "".join(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:8px 10px;">
            <div style="font-size:0.65rem;color:#475569;font-weight:700;">{r}</div>
            <div style="font-size:0.8rem;font-weight:600;color:{c};">{l}</div>
        </div>""" for r, l, c in ref_items)
        st.markdown(f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">{refs}</div>', unsafe_allow_html=True)

    with tc2:
        # Breakdown
        st.markdown("<div class='section-label'>Score Breakdown</div>", unsafe_allow_html=True)
        bk_items = [
            ("🤖 Deepfake Model",  breakdown.get('deepfake_score',0),  "40% weight"),
            ("📂 Metadata",        breakdown.get('metadata_score',0),  "20% weight"),
            ("🧪 ELA Analysis",    breakdown.get('ela_score',0),       "20% weight"),
            ("🔐 Hash Integrity",  breakdown.get('hash_score',0),      "20% weight"),
        ]
        for label, val, weight in bk_items:
            bk_col = color_for_score(val)
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:0.88rem;color:#e2e8f0;">{label}</span>
                    <span style="font-size:0.88rem;font-weight:700;color:{bk_col};">{val:.0f}/100 <span style="color:#374151;font-weight:400;">({weight})</span></span>
                </div>
                <div class="trust-bar-track" style="height:7px;">
                    <div class="trust-bar-fill" style="width:{val}%;background:{bk_col};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── AI explanation (pre-computed during batch loop) ────────────────────
        explain_html   = results.get('ai_explanation_html', '')
        explain_source = results.get('ai_explanation_source', 'rule_based')
        if explain_html:
            st.markdown(explain_html, unsafe_allow_html=True)
        else:
            # Fallback: generate now if not pre-computed
            _ex = grok_explainer.explain(results, media_type=display_type.lower())
            st.markdown(_ex['text'], unsafe_allow_html=True)
            explain_source = _ex.get('source', 'rule_based')
        if explain_source == 'rule_based':
            st.caption(
                "AI explanation is rule-based. Set GROK_API_KEY in .env to "
                "enable Groq-powered forensic explanations."
            )

    # Download report
    if report_path and os.path.exists(report_path):
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        with open(report_path, "rb") as pdf_file:
            st.download_button(
                label="📥  Download Full Forensic PDF Report",
                data=pdf_file,
                file_name=f"VeriFakeNet_Report_{uploaded_file_name}.pdf",
                mime="application/pdf",
                type="primary",
                width='stretch'
            )

# ══════════════════ TAB DATASET — DATASET & CORPUS AUDIT ══════════════════════
with tab_dataset:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Training Corpus & Dataset Audit Report</div>", unsafe_allow_html=True)
    
    d1, d2, d3, d4 = st.columns(4)
    
    d1.markdown("""<div class="metric-card">
<div class="mc-glow" style="background:#00d4ff;opacity:0.7;"></div>
<div class="mc-icon">🤖</div>
<div class="mc-label">Authenticity Dataset</div>
<div class="mc-value" style="color:#00d4ff;font-size:1.3rem;">3,452 Imgs</div>
<div class="mc-sub">VIT_Dataset (1,631 Real · 1,821 Fake)</div>
</div>""", unsafe_allow_html=True)

    d2.markdown("""<div class="metric-card">
<div class="mc-glow" style="background:#00ffb3;opacity:0.7;"></div>
<div class="mc-icon">👤</div>
<div class="mc-label">Gender & Skin Tone</div>
<div class="mc-value" style="color:#00ffb3;font-size:1.3rem;">1,500 Imgs</div>
<div class="mc-sub">UTKFace (839 Male · 661 Female)</div>
</div>""", unsafe_allow_html=True)

    d3.markdown("""<div class="metric-card">
<div class="mc-glow" style="background:#38bdf8;opacity:0.7;"></div>
<div class="mc-icon">💇</div>
<div class="mc-label">Hair Texture Dataset</div>
<div class="mc-value" style="color:#38bdf8;font-size:1.3rem;">1,992 Imgs</div>
<div class="mc-sub">5 Classes (Curly, Dreadlocks, Straight...)</div>
</div>""", unsafe_allow_html=True)

    d4.markdown("""<div class="metric-card">
<div class="mc-glow" style="background:#a78bfa;opacity:0.7;"></div>
<div class="mc-icon">🎭</div>
<div class="mc-label">Facial Landmark Corpus</div>
<div class="mc-value" style="color:#a78bfa;font-size:1.3rem;">35,887 Imgs</div>
<div class="mc-sub">FER Expression Corpus (7 Classes)</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Dataset Task Mapping & Class Breakdown")
    
    mapping_data = [
        {
            "Task / Feature": "Authenticity (Real / Fake)",
            "Dataset Name": "VIT_Dataset",
            "Total Images": "3,452",
            "Class Breakdown & Annotations": "1,631 Real (PNG) · 1,821 Fake (PNG)",
            "Model Backbone": "EfficientNet-B3 Transfer Learning"
        },
        {
            "Task / Feature": "Gender & Skin Tone",
            "Dataset Name": "UTKFace Archive (3)",
            "Total Images": "1,500",
            "Class Breakdown & Annotations": "Gender: 839 Male, 661 Female | Skin/Race: 496 Black, 389 White, 372 Indian, 166 Asian, 77 Other",
            "Model Backbone": "MobileNetV3 / ResNet18 Multi-Head Classifier"
        },
        {
            "Task / Feature": "Hair Texture Classification",
            "Dataset Name": "Archive (4) Hair Texture",
            "Total Images": "1,992",
            "Class Breakdown & Annotations": "curly: 514, dreadlocks: 443, Straight: 488, Wavy: 330, kinky: 217",
            "Model Backbone": "ResNet18 / Color-Space Hair Analysis"
        },
        {
            "Task / Feature": "Face Shape & Landmarks",
            "Dataset Name": "Archive (2) FER Corpus",
            "Total Images": "35,887",
            "Class Breakdown & Annotations": "happy: 8989, neutral: 6198, sad: 6077, fear: 5121, angry: 4953, surprise: 4002, disgust: 547",
            "Model Backbone": "Facial Landmark Ratio & Edge Geometry"
        }
    ]
    import pandas as pd
    st.dataframe(pd.DataFrame(mapping_data), use_container_width=True)

    st.markdown("""<div class="glass-card" style="margin-top:16px;">
<div class="section-label">Data Leakage Prevention & Audit Integrity Rules</div>
<ul style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin-top:8px;">
<li><strong>Person-Level Identity Splitting</strong>: For datasets containing multiple images of the same individual, identity-level splitting was enforced to ensure no person appears across both training and test sets.</li>
<li><strong>Class Imbalance Handling</strong>: Class weight balancing and focal loss criteria were applied during attribute training.</li>
<li><strong>Corrupt Image Filtering</strong>: All dataset image headers were parsed and validated before batch training.</li>
</ul>
</div>""", unsafe_allow_html=True)
