@echo off
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
if not exist .env copy .env.example .env
findstr /C:"GENERATE_A_FERNET_KEY_AND_PASTE_HERE" .env >nul
if %errorlevel%==0 (
  echo Configure ai-service\.env and replace the FACE_ENCRYPTION_KEY placeholder before starting.
  echo Generate a key with: python generate_encryption_key.py
  pause
  exit /b 1
)
findstr /C:"YOUR_PROJECT.supabase.co" .env >nul
if %errorlevel%==0 (
  echo Configure your Supabase URL and keys in ai-service\.env before starting.
  pause
  exit /b 1
)
uvicorn main:app --host 0.0.0.0 --port 8000
