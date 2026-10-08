import os
from pathlib import Path

required = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY", "FACE_ENCRYPTION_KEY"]
missing = [k for k in required if not os.getenv(k, "").strip()]
model = Path(os.getenv("LIVENESS_MODEL_PATH", "./models/face_anti_spoof.onnx"))
print("Missing required environment variables:", ", ".join(missing) if missing else "none")
print("Liveness model:", model.resolve())
print("Liveness model present:", model.exists())
raise SystemExit(1 if missing or not model.exists() else 0)
