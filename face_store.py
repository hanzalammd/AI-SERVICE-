"""Resilient face profile storage.
Tries Supabase face_profiles first; falls back to encrypted local SQLite if table is not yet created in Supabase.
"""
from __future__ import annotations
import os
import sqlite3
from pathlib import Path
from typing import Any

DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "local_face_profiles.db"

def _init_sqlite():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_profiles (
                student_id TEXT PRIMARY KEY,
                encrypted_embedding TEXT NOT NULL,
                model_name TEXT NOT NULL,
                created_by TEXT,
                consent_recorded_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()

def save_profile(admin_db: Any, payload: dict) -> None:
    _init_sqlite()
    # 1. Try Supabase
    supabase_ok = False
    if admin_db:
        try:
            admin_db.table("face_profiles").upsert(payload, on_conflict="student_id").execute()
            supabase_ok = True
        except Exception:
            supabase_ok = False

    # 2. Always persist locally as well for offline/migration resilience
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO face_profiles 
            (student_id, encrypted_embedding, model_name, created_by, consent_recorded_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            payload.get("student_id"),
            payload.get("encrypted_embedding"),
            payload.get("model_name", "buffalo_l"),
            payload.get("created_by"),
            payload.get("consent_recorded_at"),
            payload.get("updated_at")
        ))
        conn.commit()

def get_profiles(admin_db: Any, student_ids: list[str]) -> list[dict]:
    if not student_ids:
        return []
    _init_sqlite()
    profiles_map = {}

    # Try local sqlite first
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in student_ids)
            cursor = conn.execute(f"SELECT * FROM face_profiles WHERE student_id IN ({placeholders})", student_ids)
            for row in cursor.fetchall():
                profiles_map[row["student_id"]] = dict(row)
    except Exception:
        pass

    # Try Supabase to overlay or fetch newer cloud profiles
    if admin_db:
        try:
            data = admin_db.table("face_profiles").select("student_id,encrypted_embedding,model_name,consent_recorded_at,updated_at").in_("student_id", student_ids).execute().data or []
            for p in data:
                if p.get("encrypted_embedding"):
                    profiles_map[p["student_id"]] = p
        except Exception:
            pass

    return list(profiles_map.values())

def delete_profile(admin_db: Any, student_id: str) -> None:
    _init_sqlite()
    if admin_db:
        try:
            admin_db.table("face_profiles").delete().eq("student_id", student_id).execute()
        except Exception:
            pass
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM face_profiles WHERE student_id = ?", (student_id,))
            conn.commit()
    except Exception:
        pass
