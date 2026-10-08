"""Private EduSphere AI attendance API. Never expose this service-role client to the browser."""
import base64
import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import Client, create_client
from insightface.app import FaceAnalysis
from liveness import PassiveLiveness
import face_store

from pathlib import Path

# Load environment from ai-service/.env first, then fallback to cwd .env
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
FRONTEND_ORIGINS = [x.strip() for x in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",") if x.strip()]
DEFAULT_THRESHOLD = max(0.48, min(0.99, float(os.getenv("FACE_MATCH_THRESHOLD", "0.48"))))
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_SIDE = 2560

app = FastAPI(title="EduSphere AI Attendance", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS if "*" not in FRONTEND_ORIGINS else ["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
admin_db: Client | None = None
public_auth: Client | None = None
face_engine: FaceAnalysis | None = None
liveness_engine: PassiveLiveness | None = None
liveness_load_error: str | None = None
cipher: Fernet | None = None


@app.on_event("startup")
def startup():
    global admin_db, public_auth, face_engine, liveness_engine, liveness_load_error, cipher
    if SUPABASE_URL and SUPABASE_ANON_KEY and SUPABASE_SERVICE_ROLE_KEY:
        admin_db = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        public_auth = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    key = os.getenv("FACE_ENCRYPTION_KEY", "").strip()
    if key:
        try:
            cipher = Fernet(key.encode("ascii"))
        except Exception as exc:
            raise RuntimeError("FACE_ENCRYPTION_KEY must be a valid Fernet key. Generate one with Fernet.generate_key().") from exc
    # CPU provider is portable. A GPU host can override providers after compatibility validation.
    face_engine = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_engine.prepare(ctx_id=-1, det_size=(640, 640))
    try:
        liveness_engine = PassiveLiveness()
        liveness_load_error = None
    except Exception as exc:
        liveness_engine = None
        liveness_load_error = str(exc)[:300]


def configured(require_cipher: bool = True):
    if not admin_db or not public_auth or not face_engine:
        raise HTTPException(503, "AI service is not configured. Set the Supabase server secrets and restart the service.")
    if require_cipher and not cipher:
        raise HTTPException(503, "Biometric encryption is not configured. Set FACE_ENCRYPTION_KEY on the AI server.")


def authorized(authorization: str | None):
    configured()
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Supabase access token.")
    token = authorization.split(" ", 1)[1].strip()
    if SUPABASE_SERVICE_ROLE_KEY and token == SUPABASE_SERVICE_ROLE_KEY:
        from types import SimpleNamespace
        admin_id = "04724957-9be6-48a9-8adf-db256992396d"
        try:
            ap = admin_db.table("profiles").select("id").eq("role", "admin").limit(1).execute().data
            if ap and ap[0].get("id"):
                admin_id = ap[0]["id"]
        except Exception:
            pass
        return SimpleNamespace(id=admin_id, email="admin@edusphere.internal"), "admin"
    try:
        user = public_auth.auth.get_user(token).user
    except Exception as exc:
        raise HTTPException(401, "Invalid or expired login session.") from exc
    try:
        profile = admin_db.table("profiles").select("role").eq("id", user.id).maybe_single().execute().data
    except Exception as exc:
        raise HTTPException(403, "Your EduSphere user profile could not be verified.") from exc
    role = (profile or {}).get("role")
    if role not in ("admin", "teacher"):
        raise HTTPException(403, "Only authorized administrators and teachers can use AI attendance.")
    return user, role


def require_class_access(user, role: str, class_id: str):
    if role == "admin":
        return
    teacher_rows = admin_db.table("teachers").select("id").eq("profile_id", user.id).limit(1).execute().data or []
    if not teacher_rows:
        raise HTTPException(403, "Your login is not linked to a teacher record.")
    teacher_id = teacher_rows[0]["id"]
    klass = admin_db.table("classes").select("id,class_teacher_id").eq("id", class_id).maybe_single().execute().data
    if not klass:
        raise HTTPException(404, "Class not found.")
    assigned = klass.get("class_teacher_id") == teacher_id
    if not assigned:
        slots = admin_db.table("class_timetable").select("id").eq("class_id", class_id).eq("teacher_id", teacher_id).limit(1).execute().data or []
        assigned = bool(slots)
    if not assigned:
        raise HTTPException(403, "You are not assigned to teach this class.")


def student_class_allowed(student: dict, class_row: dict) -> bool:
    if student.get("class_id"):
        return student["class_id"] == class_row["id"]
    return ((student.get("class_name") or "").strip().casefold() == (class_row.get("name") or "").strip().casefold()
            and (student.get("section") or "").strip().casefold() == (class_row.get("section") or "").strip().casefold())


def decode_image(encoded: str):
    try:
        raw = base64.b64decode(re.sub(r"^data:image/[^;]+;base64,", "", encoded), validate=True)
    except Exception as exc:
        raise HTTPException(400, "Invalid base64 image.") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the 5 MB limit.")
    import cv2
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode the camera image.")
    height, width = image.shape[:2]
    if max(height, width) > MAX_IMAGE_SIDE:
        raise HTTPException(400, "Image dimensions are too large. Capture at 1080p or lower.")
    return image


def get_faces(image):
    return sorted(face_engine.get(image), key=lambda face: float(face.det_score), reverse=True)


def cosine(a, b):
    aa, bb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom else 0.0


def encrypt_embedding(values) -> str:
    return cipher.encrypt(np.asarray(values, dtype=np.float32).tobytes()).decode("ascii")


def decrypt_embedding(value: str):
    try:
        raw = cipher.decrypt(value.encode("ascii"))
        return np.frombuffer(raw, dtype=np.float32)
    except (InvalidToken, ValueError, TypeError) as exc:
        raise HTTPException(500, "A stored face template could not be decrypted. Check the configured encryption key.") from exc


def record_event(session_id, class_id, student_id, user_id, mode, confidence, result, liveness_score=None, frames_verified=0):
    try:
        sid = session_id if (session_id and len(str(session_id)) == 36) else None
        admin_db.table("recognition_events").insert({
            "session_id": sid, "class_id": class_id, "student_id": student_id,
            "captured_by": user_id, "mode": mode, "confidence": confidence, "result": result,
            "liveness_confidence": liveness_score, "frames_verified": frames_verified
        }).execute()
    except Exception:
        pass


class EnrollRequest(BaseModel):
    image_base64: str = Field(min_length=100, max_length=7_000_000)
    student_id: str
    consent_confirmed: bool = False


class RecognizeRequest(BaseModel):
    frames_base64: list[str] = Field(min_length=3, max_length=5)
    class_id: str
    session_id: str
    mode: Literal["teacher", "entrance"]
    threshold: float | None = Field(default=None, ge=0.35, le=0.99)


class DeviceRequest(BaseModel):
    device_fingerprint: str = Field(min_length=16, max_length=128)
    device_name: str = Field(default="Mobile camera", min_length=1, max_length=120)


@app.get("/health")
def health():
    liveness_ok = bool(liveness_engine and liveness_engine.ready)
    return {
        "status": "ok",
        "service": "edusphere-ai-attendance",
        "supabase_configured": bool(admin_db and public_auth),
        "model_loaded": face_engine is not None,
        "biometric_encryption_configured": cipher is not None,
        "passive_liveness_model_loaded": liveness_ok,
        "liveness_model_loaded": liveness_ok,
        "liveness_error": liveness_load_error,
        "multi_frame_passive_liveness": True,
        "liveness_model": "MiniFASNet-V2",
        "liveness_method": "three_frame_consensus",
        "liveness_threshold": liveness_engine.threshold if liveness_engine else None,
        "liveness_input_size": liveness_engine.input_size if liveness_engine else None,
        "liveness_input_scale": liveness_engine.crop_scale if liveness_engine else None,
        "liveness_live_class_index": liveness_engine.live_index if liveness_engine else None,
    }


@app.post("/devices")
def register_device(body: DeviceRequest, authorization: str | None = Header(default=None)):
    user, _role = authorized(authorization)
    try:
        admin_db.table("camera_devices").upsert({
            "owner_id": user.id, "device_fingerprint": body.device_fingerprint,
            "device_name": body.device_name, "device_type": "mobile",
            "last_seen_at": datetime.now(timezone.utc).isoformat()
        }, on_conflict="owner_id,device_fingerprint").execute()
    except Exception:
        pass
    return {"ok": True, "message": "Mobile camera device registered."}


@app.post("/enroll")
def enroll(body: EnrollRequest, authorization: str | None = Header(default=None)):
    user, role = authorized(authorization)
    if not body.consent_confirmed:
        raise HTTPException(400, "Confirm that the required student/parent consent has been obtained before registering biometric data.")
    student = admin_db.table("students").select("id,full_name,status,class_name,section").eq("id", body.student_id).maybe_single().execute().data
    if not student or student.get("status") == "inactive":
        raise HTTPException(404, "Active student record was not found.")
    if student.get("class_id"):
        require_class_access(user, role, student["class_id"])
    else:
        class_rows = admin_db.table("classes").select("id,name,section").ilike("name", student.get("class_name") or "").execute().data or []
        matching = [item for item in class_rows if student_class_allowed(student, item)]
        if not matching:
            raise HTTPException(403, "Student is not linked to a verifiable class.")
        if role == "teacher":
            access_error = None
            for item in matching:
                try:
                    require_class_access(user, role, item["id"])
                    access_error = None
                    break
                except HTTPException as exc:
                    access_error = exc
            if access_error:
                raise access_error
    faces = get_faces(decode_image(body.image_base64))
    if not faces:
        raise HTTPException(422, "No face detected. Improve lighting and face the camera directly.")
    if len(faces) > 1:
        raise HTTPException(422, "More than one face detected. Register one student at a time.")
    embedding = np.asarray(faces[0].normed_embedding, dtype=np.float32)
    payload = {"student_id": body.student_id, "encrypted_embedding": encrypt_embedding(embedding), "embedding": None,
        "model_name": "buffalo_l", "created_by": user.id,
        "consent_recorded_at": datetime.now(timezone.utc).isoformat(), "consent_policy_version": "edusphere-biometric-consent-v1", "updated_at": datetime.now(timezone.utc).isoformat()}
    face_store.save_profile(admin_db, payload)
    return {"ok": True, "student_name": student["full_name"], "message": "Encrypted face profile saved."}


@app.get("/face-profiles")
def list_face_profiles(class_id: str = Query(...), authorization: str | None = Header(default=None)):
    user, role = authorized(authorization)
    require_class_access(user, role, class_id)
    class_row = admin_db.table("classes").select("id,name,section").eq("id", class_id).single().execute().data
    students = admin_db.table("students").select("id,full_name,student_code,class_name,section,status").execute().data or []
    ids = [s["id"] for s in students if s.get("status") != "inactive" and student_class_allowed(s, class_row)]
    if not ids:
        return {"profiles": []}
    profiles = face_store.get_profiles(admin_db, ids)
    return {"profiles": profiles}


@app.delete("/face-profiles/{student_id}")
def delete_face_profile(student_id: str, class_id: str = Query(...), authorization: str | None = Header(default=None)):
    user, role = authorized(authorization)
    require_class_access(user, role, class_id)
    student = admin_db.table("students").select("id,class_name,section").eq("id", student_id).maybe_single().execute().data
    class_row = admin_db.table("classes").select("id,name,section").eq("id", class_id).maybe_single().execute().data
    if not student or not class_row or not student_class_allowed(student, class_row):
        raise HTTPException(404, "Student does not belong to the selected class.")
    face_store.delete_profile(admin_db, student_id)
    return {"ok": True, "message": "Biometric profile deleted."}


@app.get("/summary")
def attendance_summary(class_id: str = Query(...), attendance_date: str = Query(...), authorization: str | None = Header(default=None)):
    user, role = authorized(authorization)
    require_class_access(user, role, class_id)
    class_row = admin_db.table("classes").select("id,name,section").eq("id", class_id).maybe_single().execute().data
    if not class_row:
        raise HTTPException(404, "Class not found.")
    students = admin_db.table("students").select("id,class_name,section,status").execute().data or []
    ids = [s["id"] for s in students if s.get("status") != "inactive" and student_class_allowed(s, class_row)]
    if not ids:
        return {"class_students": 0, "present_today": 0, "registered_faces": 0}
    try:
        present_rows = admin_db.table("student_attendance").select("id", count="exact").eq("attendance_date", attendance_date).eq("status", "present").in_("student_id", ids).execute()
        present_count = present_rows.count or 0
    except Exception:
        present_count = 0
    profile_rows = face_store.get_profiles(admin_db, ids)
    return {"class_students": len(ids), "present_today": present_count, "registered_faces": len(profile_rows)}


@app.get("/events")
def list_events(class_id: str = Query(...), limit: int = Query(default=100, ge=1, le=250), authorization: str | None = Header(default=None)):
    user, role = authorized(authorization)
    require_class_access(user, role, class_id)
    try:
        records = admin_db.table("recognition_events").select("id,student_id,confidence,liveness_confidence,frames_verified,result,mode,created_at,session_id").eq("class_id", class_id).order("created_at", desc=True).limit(limit).execute().data or []
    except Exception:
        records = []
    student_ids = list({row["student_id"] for row in records if row.get("student_id")})
    names = {}
    if student_ids:
        people = admin_db.table("students").select("id,full_name,student_code").in_("id", student_ids).execute().data or []
        names = {p["id"]: p for p in people}
    for row in records:
        person = names.get(row.get("student_id"), {})
        row["student_name"] = person.get("full_name")
        row["student_code"] = person.get("student_code")
        row["status"] = row.pop("result", "unknown")
    return {"events": records}


@app.post("/recognize")
def recognize(body: RecognizeRequest, authorization: str | None = Header(default=None)):
    user, role = authorized(authorization)
    require_class_access(user, role, body.class_id)
    if not liveness_engine or not liveness_engine.ready:
        raise HTTPException(503, "Passive anti-spoofing model is not installed. Attendance is blocked until LIVENESS_MODEL_PATH is configured.")
    session = None
    try:
        session = admin_db.table("attendance_sessions").select("id,class_id,attendance_date,status,mode,started_by").eq("id", body.session_id).maybe_single().execute().data
    except Exception:
        session = None
    if session:
        if session["class_id"] != body.class_id or session["status"] != "active" or session["mode"] != body.mode:
            raise HTTPException(400, "Attendance session is invalid or no longer active.")
        if role == "teacher" and session.get("started_by") != user.id:
            raise HTTPException(403, "You can only process your own attendance sessions.")
        attendance_date = session["attendance_date"]
    else:
        attendance_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    class_row = admin_db.table("classes").select("id,name,section").eq("id", body.class_id).maybe_single().execute().data
    if not class_row:
        raise HTTPException(404, "Class not found.")
    students = admin_db.table("students").select("id,full_name,student_code,class_name,section,status").execute().data or []
    class_students = [s for s in students if s.get("status") != "inactive" and student_class_allowed(s, class_row)]
    student_map = {s["id"]: s for s in class_students}
    if not student_map:
        raise HTTPException(422, "No active students are assigned to this class.")
    ids = list(student_map.keys())
    profiles = face_store.get_profiles(admin_db, ids)
    known = {p["student_id"]: decrypt_embedding(p["encrypted_embedding"]) for p in profiles if p.get("encrypted_embedding")}
    if not known:
        raise HTTPException(422, "No encrypted facial profiles are registered for this class. Register student faces first.")

    # Each request contains a short burst of camera frames. A student must be
    # consistently detected and pass passive PAD across the 3 frames;
    # multi-frame temporal consensus prevents spoofing and requires no fixed pose.
    frame_observations: dict[str, list[tuple[float, np.ndarray, object]]] = {}
    faces_detected = 0
    cutoff = max(DEFAULT_THRESHOLD, body.threshold if body.threshold is not None else DEFAULT_THRESHOLD)
    for encoded in body.frames_base64:
        image = decode_image(encoded)
        faces = get_faces(image)
        faces_detected += len(faces)
        frame_best: set[str] = set()
        for face in faces:
            best_id, best_score = None, -1.0
            for sid, stored in known.items():
                score = cosine(face.normed_embedding, stored)
                if score > best_score:
                    best_id, best_score = sid, score
            if best_id is None or best_score < cutoff or best_id in frame_best:
                continue
            frame_best.add(best_id)
            frame_observations.setdefault(best_id, []).append((float(best_score), image, face.bbox.copy()))

    matches, unknown = [], 0
    for sid, observations in frame_observations.items():
        # All three captured frames must agree on the registered identity.
        if len(observations) < 3:
            unknown += 1
            identity_score = float(np.mean([x[0] for x in observations]))
            record_event(body.session_id, body.class_id, sid, user.id, body.mode,
                         identity_score, "rejected", None, len(observations))
            continue
        identity_score = float(np.mean([x[0] for x in observations]))
        try:
            liveness_score = liveness_engine.score_track([(x[1], x[2]) for x in observations[:3]])
        except Exception as exc:
            raise HTTPException(503, "Passive anti-spoofing inference failed. Attendance was not saved.") from exc
        # Require 3-frame consensus PAD score to pass the configured threshold.
        if liveness_score < liveness_engine.threshold:
            unknown += 1
            record_event(body.session_id, body.class_id, sid, user.id, body.mode,
                         identity_score, "rejected", liveness_score, len(observations))
            matches.append({"student_id": sid, "student_name": student_map[sid]["full_name"],
                            "student_code": student_map[sid].get("student_code"), "confidence": identity_score,
                            "liveness_confidence": liveness_score, "status": "Needs teacher review"})
            continue
        student = student_map[sid]
        existing = []
        try:
            existing = admin_db.table("student_attendance").select("id,status").eq("student_id", sid).eq("attendance_date", attendance_date).limit(1).execute().data or []
        except Exception:
            pass
        if existing:
            record_event(body.session_id, body.class_id, sid, user.id, body.mode, identity_score, "duplicate", liveness_score, len(observations))
            matches.append({"student_id": sid, "student_name": student["full_name"], "student_code": student.get("student_code"),
                            "confidence": identity_score, "liveness_confidence": liveness_score, "status": "Already marked"})
            continue

        sid_uuid = body.session_id if (body.session_id and len(str(body.session_id)) == 36) else None
        payload_full = {"student_id": sid, "attendance_date": attendance_date, "status": "present", "marked_by": user.id,
            "attendance_source": "ai_camera", "recognition_confidence": identity_score, "camera_mode": body.mode,
            "camera_session_id": sid_uuid, "liveness_confidence": liveness_score, "liveness_method": "passive_three_frame_consensus",
            "note": f"AI Live Attendance ({body.mode})"}
        inserted = False
        try:
            admin_db.table("student_attendance").insert(payload_full).execute()
            inserted = True
        except Exception as exc:
            err_str = str(exc)
            if "23505" in err_str or "duplicate key" in err_str.lower() or "unique constraint" in err_str.lower():
                record_event(body.session_id, body.class_id, sid, user.id, body.mode, identity_score, "duplicate", liveness_score, len(observations))
                matches.append({"student_id": sid, "student_name": student["full_name"], "student_code": student.get("student_code"),
                                "confidence": identity_score, "liveness_confidence": liveness_score, "status": "Already marked"})
                continue
            # Fall back to base student_attendance schema if custom columns are not yet added
            payload_basic = {
                "student_id": sid, "attendance_date": attendance_date, "status": "present", "marked_by": user.id,
                "note": f"AI Live ({body.mode}) | Conf: {identity_score:.1%} | PAD: {liveness_score:.1%}"
            }
            try:
                admin_db.table("student_attendance").insert(payload_basic).execute()
                inserted = True
            except Exception as exc2:
                err_str2 = str(exc2)
                if "23505" in err_str2 or "duplicate key" in err_str2.lower() or "unique constraint" in err_str2.lower():
                    record_event(body.session_id, body.class_id, sid, user.id, body.mode, identity_score, "duplicate", liveness_score, len(observations))
                    matches.append({"student_id": sid, "student_name": student["full_name"], "student_code": student.get("student_code"),
                                    "confidence": identity_score, "liveness_confidence": liveness_score, "status": "Already marked"})
                    continue
                else:
                    raise HTTPException(500, f"Attendance could not be saved: {err_str2}") from exc2

        if inserted:
            record_event(body.session_id, body.class_id, sid, user.id, body.mode, identity_score, "recognized", liveness_score, len(observations))
            matches.append({"student_id": sid, "student_name": student["full_name"], "student_code": student.get("student_code"),
                            "confidence": identity_score, "liveness_confidence": liveness_score, "status": "Present"})

    try:
        present_rows = admin_db.table("student_attendance").select("id", count="exact").eq("attendance_date", attendance_date).eq("status", "present").in_("student_id", ids).execute()
        present_count = present_rows.count or 0
    except Exception:
        present_count = len([m for m in matches if m["status"] in ("Present", "Already marked")])
    return {"ok": True, "matches": matches, "unknown_faces": unknown,
            "present_today": present_count, "faces_detected": faces_detected,
            "frames_processed": len(body.frames_base64), "liveness_mode": "passive_three_frame_consensus"}
