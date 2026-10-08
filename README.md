# EduSphere AI Attendance API

Private Python service for mobile-camera student face registration and attendance recognition. It runs separately from the Vite frontend. Never expose the Supabase service-role key or face-encryption key to the browser.

## Prerequisites

- Python 3.10 or 3.11 (recommended for the pinned InsightFace/ONNX dependencies).
- Existing EduSphere Supabase project.
- Run `supabase/07_AI_MOBILE_ATTENDANCE.sql` and then `supabase/08_PASSIVE_MOBILE_LIVENESS.sql` from the project root in Supabase SQL Editor.
- Obtain and document the appropriate student/parent consent before collecting biometric data.

## Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with the Supabase project URL, anon/publishable key, service-role key, exact frontend origin(s), and a persistent Fernet encryption key. Place a compatible, properly licensed ONNX PAD model at `ai-service/models/face_anti_spoof.onnx` (or set a different absolute `LIVENESS_MODEL_PATH`). For Docker Compose, the `models` directory is mounted read-only into `/app/models`. Generate the key once:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Keep `FACE_ENCRYPTION_KEY` backed up in a password manager/secret manager. Do not rotate or lose it without a planned decrypt-and-re-encrypt migration; otherwise enrolled templates cannot be read. Then start the service:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Set the frontend's `VITE_AI_SERVICE_URL=http://localhost:8000` and restart Vite. Phone camera access requires HTTPS (localhost is allowed for local development).

## API endpoints

- `GET /health` — service/model/database/encryption readiness; contains no secrets.
- `POST /devices` — register the signed-in teacher's mobile browser/device.
- `POST /enroll` — enroll one face; requires student ID and explicit consent confirmation.
- `GET /face-profiles?class_id=...` — list enrollment metadata only, never embeddings.
- `DELETE /face-profiles/{student_id}?class_id=...` — delete a student's biometric template after consent withdrawal/retention expiry.
- `GET /events?class_id=...&limit=100` — fetch class-scoped recognition event history.
- `POST /recognize` — process a burst of three to five still frames for multi-face identity matching and passive liveness verification. No blink/head-turn challenge is requested.

All endpoints except `/health` require `Authorization: Bearer <Supabase access token>`. The API verifies the token, role, and teacher-to-class assignment itself because it uses a service-role database connection. Teachers can only access classes assigned through class teacher or timetable records.

## Deployment

Build from this directory using the included Dockerfile, or deploy the Docker image to a private Python container host. Configure all `.env` values as server-side secrets. Set `FRONTEND_ORIGINS` to the exact HTTPS origin(s) serving EduSphere. Set `VITE_AI_SERVICE_URL` to the public HTTPS endpoint of this service. Put the service behind HTTPS, a reverse proxy/API gateway, request-size limits, rate limiting, monitoring and backups. Do not expose the service-role key.

## Important limitations and privacy

- The browser sends short bursts of still JPEG frames only; it does not stream or persist classroom video. Mobile live attendance captures a three-frame burst about every five seconds while the page remains open and the screen awake.
- Facial embeddings are encrypted with Fernet before storage in `face_profiles.encrypted_embedding`. The private server is the only component holding the decryption key. The database migration clears any legacy unencrypted embedding arrays; those students must be enrolled again.
- The default similarity threshold is only a starting point. The server enforces its configured minimum; raising the threshold is allowed. Calibrate with representative consented samples and measure false-accept/false-reject rates before using this for official attendance.
- Attendance is blocked unless the MiniFASNet-V2 ONNX anti-spoofing model is configured with `LIVENESS_MODEL_PATH`. The adapter expects a BGR float32 image tensor `[1,3,80,80]` in the `[0,1]` range and the documented three output classes `[live, print-attack, replay-attack]`. Each of three consecutive camera frames is checked independently; the lowest live probability must pass. This is three-frame consensus, not a temporal neural network. Download the model with `download_liveness_model.ps1`, which verifies the published SHA-256.
- Each scan uses three short frames. A student must be matched in all three captured frames and pass the configured passive PAD threshold before attendance is inserted. This reduces simple photo/screen presentation attacks but is not a guarantee against high-quality replay, deepfake, camera-feed injection, or other attacks. Test false-accept and false-reject rates using consented representative samples.
- The repository does not include pretrained anti-spoofing weights. Obtain a model with suitable licensing, verify its input/output contract, configure the environment values, then confirm `/health` reports `passive_liveness_model_loaded: true`. Until then, AI attendance intentionally remains disabled; manual attendance continues to work.
- Set a retention/deletion policy, document consent, limit staff access, and provide a non-biometric alternative. Raw camera frames are processed in memory and are not stored by this service.
- Attendance continues to use EduSphere's existing unique `(student_id, attendance_date)` rule, so manual and AI marking cannot create duplicate daily records.
- InsightFace model weights are downloaded/initialized at first startup. Review InsightFace/model licensing and local biometric/privacy requirements before institutional or commercial deployment.


### Download the passive liveness model (Windows)

From this `ai-service` directory, run:

`powershell -ExecutionPolicy Bypass -File .\download_liveness_model.ps1`

The script stores `models/face_anti_spoof.onnx` and checks the model SHA-256. The model is licensed Apache-2.0; see `models/README.md` for attribution and limitations.
