@echo off
cd /d "%~dp0"
if exist "VeriFakeNet" (
    cd VeriFakeNet
)
echo Starting VeriFakeNet Face Authenticity & Attribute System...
C:\Users\Dell\AppData\Local\Programs\Python\Python312\python.exe run_all.py
pause
