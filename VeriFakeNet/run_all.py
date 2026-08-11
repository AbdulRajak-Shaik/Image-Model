import os
import sys
import subprocess

def run_everything():
    python_exe = r"C:\Users\Dell\AppData\Local\Programs\Python\Python312\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    print("=" * 60)
    print("🛡️ VeriFakeNet: Running Complete System Pipeline")
    print("=" * 60)

    # 1. Dataset Audit
    print("\n[1/3] Running Dataset Audit & Validation...")
    subprocess.run([python_exe, "preprocessing/dataset_validation.py"])

    # 2. Pipeline Test
    print("\n[2/3] Running Integrated Model Pipeline Test...")
    subprocess.run([python_exe, "scratch/test_full_pipeline.py"])

    # 3. Launch Streamlit UI
    print("\n[3/3] Launching Streamlit Web Application...")
    print("Opening VeriFakeNet UI on http://localhost:8501 ...")
    try:
        subprocess.run([python_exe, "-m", "streamlit", "run", "streamlit_app/app.py"])
    except KeyboardInterrupt:
        print("\n🛡️ [VeriFakeNet] Streamlit web server closed by user cleanly.")
        sys.exit(0)

if __name__ == "__main__":
    try:
        run_everything()
    except KeyboardInterrupt:
        print("\n🛡️ Exited VeriFakeNet pipeline.")
        sys.exit(0)
