import os
import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from pptx import Presentation
except Exception:
    Presentation = None


APP_VERSION = "5.0"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None

CHUNK_CHARS = int(os.getenv("AI_CHUNK_CHARS", "24000"))
MAX_RETRIES = 3

app = FastAPI(title="Study AI Backend", version=APP_VERSION)

origins = [
    x.strip()
    for x in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional study material. AI chat does NOT depend on this.
last_material = {
    "name": None,
    "text": "",
    "image_data": None,
    "mime": None,
}

SYSTEM_GENERAL = """You are Study AI, a helpful academic assistant.
Answer the user's question directly even when no file is uploaded.
You can answer general, study, medical and pharmaceutical education questions.
For medical/pharmaceutical topics, use careful educational language and do not invent facts.
If the user asks for diagnosis, treatment, dosing, or an urgent medical decision, explain that the answer is educational and advise consulting a qualified clinician when appropriate.
Use clear Arabic when the user writes Arabic, and preserve useful English medical/pharmaceutical terms beside Arabic terms.
Do not claim an exam question is guaranteed to appear.
"""

SYSTEM_WITH_MATERIAL = """You are Study AI, a medical and pharmaceutical study assistant.
The attached material is optional reference material, not a requirement for answering.
Use it when relevant. If the answer is not contained in the material, you may use general knowledge and clearly label information added from outside the material as: "معلومة إضافية".
Never invent scientific facts, doses, drug names, numbers, units, abbreviations, or exam certainty.
Use accurate Arabic medical/pharmaceutical terminology and keep important English terms beside it.
"""

class ChatRequest(BaseModel):
    message: str
    topic: Optional[str] = None

class FeatureRequest(BaseModel):
    feature: str
    message: Optional[str] = None
    topic: Optional[str] = None
    material: Optional[str] = None


def require_client():
    if not client:
        raise HTTPException(503, "OPENAI_API_KEY غير مضبوط على الخادم.")


def extract_text(data: bytes, name: str) -> str:
    ext = Path(name).suffix.lower()

    if ext == ".pdf" and PdfReader:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx" and Document:
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if ext == ".pptx" and Presentation:
        prs = Presentation(io.BytesIO(data))
        parts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text)
        return "\n".join(parts)

    if ext in [".txt", ".md"]:
        return data.decode("utf-8", errors="replace")

    return ""


def _responses_create(**kwargs):
    require_client()
    delay = 2.0

    for attempt in range(MAX_RETRIES):
        try:
            return client.responses.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            retryable = any(
                token in msg
                for token in ("429", "rate_limit", "timeout", "temporarily unavailable")
            )
            if not retryable or attempt >= MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2


def call_ai(
    instruction: str,
    system: str = SYSTEM_GENERAL,
    max_output_tokens: int = 1800,
    image_data: Optional[str] = None,
    image_mime: Optional[str] = None,
) -> str:
    require_client()

    if image_data:
        input_value = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction},
                {
                    "type": "input_image",
                    "image_url": f"data:{image_mime};base64,{image_data}",
                },
            ],
        }]
    else:
        input_value = instruction

    response = _responses_create(
        model=MODEL,
        instructions=system,
        input=input_value,
        max_output_tokens=max_output_tokens,
    )
    return (response.output_text or "").strip()


def ask_with_optional_material(
    instruction: str,
    material: Optional[str] = None,
    max_output_tokens: int = 1800,
) -> str:
    if material and material.strip():
        material = material[:CHUNK_CHARS]
        prompt = (
            instruction
            + "\n\nSTUDY MATERIAL REFERENCE:\n"
            + material
        )
        return call_ai(
            prompt,
            system=SYSTEM_WITH_MATERIAL,
            max_output_tokens=max_output_tokens,
        )
    return call_ai(
        instruction,
        system=SYSTEM_GENERAL,
        max_output_tokens=max_output_tokens,
    )


@app.get("/")
def root():
    return {
        "name": "Study AI Backend",
        "version": APP_VERSION,
        "status": "online",
        "ai_configured": bool(client),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL,
        "ai_configured": bool(client),
        "material_loaded": bool(last_material["name"]),
    }


@app.get("/ai-test")
@app.post("/ai-test")
def ai_test():
    """
    Real AI connectivity test used by the Android status button.
    Both GET and POST are supported so older/newer app builds work.
    """
    try:
        answer = call_ai(
            "Reply with exactly: Study AI is connected.",
            system="You are a connectivity test. Return only the requested short sentence.",
            max_output_tokens=40,
        )
        return {
            "ok": True,
            "available": True,
            "model": MODEL,
            "message": answer or "Study AI is connected.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "available": False,
            "model": MODEL,
            "message": str(exc)[:500],
        }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Independent AI chat. No uploaded file is required.
    """
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(400, "اكتب سؤالك أولاً.")

    topic = (req.topic or "").strip()
    context = f"\nTopic: {topic}" if topic else ""

    answer = call_ai(
        "Answer the user's question directly.\n"
        f"User question:\n{message}{context}",
        system=SYSTEM_GENERAL,
        max_output_tokens=2200,
    )
    return {
        "ok": True,
        "answer": answer,
        "model": MODEL,
        "file_required": False,
    }


@app.post("/chat-file")
async def chat_file(
    message: str = "",
    file: UploadFile | None = File(default=None),
):
    """
    AI chat with an optional file. The file is a reference only.
    """
    message = (message or "").strip()
    if not message:
        raise HTTPException(400, "اكتب سؤالك أولاً.")

    material = None
    filename = None

    if file is not None:
        data = await file.read()
        filename = file.filename
        mime = file.content_type or ""

        if mime.startswith("image/"):
            answer = call_ai(
                f"Answer this user question using the attached image when relevant:\n{message}",
                system=SYSTEM_WITH_MATERIAL,
                max_output_tokens=2200,
                image_data=base64.b64encode(data).decode("ascii"),
                image_mime=mime,
            )
        else:
            material = extract_text(data, file.filename or "file")
            if not material:
                raise HTTPException(
                    400,
                    "لم أستطع استخراج النص من الملف. استخدم PDF/DOCX/PPTX أو صورة واضحة.",
                )
            answer = ask_with_optional_material(
                message,
                material=material,
                max_output_tokens=2200,
            )
    else:
        answer = call_ai(message, system=SYSTEM_GENERAL, max_output_tokens=2200)

    return {
        "ok": True,
        "answer": answer,
        "model": MODEL,
        "file_required": False,
        "file_used": bool(filename),
        "filename": filename,
    }


@app.post("/feature")
def feature(req: FeatureRequest):
    """
    Shared optional-context endpoint for Study Plan, Questions,
    Flashcards, Terms, Quick Review, Rewrite and similar screens.
    A topic/message is enough; material is optional.
    """
    feature = (req.feature or "study").strip()
    topic = (req.topic or "").strip()
    message = (req.message or "").strip()

    request_text = (
        f"Feature: {feature}\n"
        f"Topic: {topic}\n"
        f"User request: {message}\n\n"
        "Generate the requested study content in a clear, useful format. "
        "If a topic is provided without a file, answer from general knowledge. "
        "If material is provided, use it as reference."
    )

    answer = ask_with_optional_material(
        request_text,
        material=req.material,
        max_output_tokens=2600,
    )

    return {
        "ok": True,
        "feature": feature,
        "answer": answer,
        "model": MODEL,
        "file_required": False,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    mime = file.content_type or ""
    filename = file.filename or "study_file"

    text = extract_text(data, filename)

    if not text and not mime.startswith("image/"):
        raise HTTPException(
            400,
            "لم أستطع استخراج النص من هذا الملف. استخدم PDF/DOCX/PPTX أو صورة واضحة.",
        )

    last_material["name"] = filename
    last_material["text"] = text
    last_material["mime"] = mime
    last_material["image_data"] = (
        base64.b64encode(data).decode("ascii")
        if mime.startswith("image/")
        else None
    )

    prompt = """حلّل المادة الدراسية تحليلاً شاملاً، وأعد JSON صالح فقط بدون Markdown.
المفاتيح:
title, summary, difficulty, understanding, high_yield, topics,
must_memorize, must_understand, terms, comparisons, rewrite,
study_plan, quick_revision, questions, flashcards.

terms عناصرها: english, arabic, explanation.
questions عناصرها: question, options, answer, explanation.
flashcards عناصرها: front, back.
understanding رقم من 0 إلى 100 يعبّر عن وضوح المادة وبنيتها وليس عن مستوى الطالب.
لا تدّعِ أن سؤالاً مضمون في الامتحان.
"""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "difficulty": {"type": "string"},
            "understanding": {"type": "integer", "minimum": 0, "maximum": 100},
            "high_yield": {"type": "array", "items": {"type": "string"}},
            "topics": {"type": "array", "items": {"type": "string"}},
            "must_memorize": {"type": "array", "items": {"type": "string"}},
            "must_understand": {"type": "array", "items": {"type": "string"}},
            "terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "english": {"type": "string"},
                        "arabic": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["english", "arabic", "explanation"],
                },
            },
            "comparisons": {"type": "array", "items": {"type": "string"}},
            "rewrite": {"type": "string"},
            "study_plan": {"type": "array", "items": {"type": "string"}},
            "quick_revision": {"type": "array", "items": {"type": "string"}},
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "answer": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["question", "options", "answer", "explanation"],
                },
            },
            "flashcards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "front": {"type": "string"},
                        "back": {"type": "string"},
                    },
                    "required": ["front", "back"],
                },
            },
        },
        "required": [
            "title", "summary", "difficulty", "understanding", "high_yield",
            "topics", "must_memorize", "must_understand", "terms",
            "comparisons", "rewrite", "study_plan", "quick_revision",
            "questions", "flashcards",
        ],
    }

    if mime.startswith("image/"):
        raw = call_ai(
            prompt,
            system=SYSTEM_WITH_MATERIAL,
            max_output_tokens=5000,
            image_data=last_material["image_data"],
            image_mime=mime,
        )
    else:
        excerpt = text[:CHUNK_CHARS]
        raw = call_ai(
            prompt + "\n\nSTUDY MATERIAL:\n" + excerpt,
            system=SYSTEM_WITH_MATERIAL,
            max_output_tokens=5000,
        )

    try:
        # A second structured request is intentionally avoided here; the
        # current Responses API JSON-schema path is used directly below.
        if mime.startswith("image/"):
            pass
        # Re-run with schema for deterministic JSON when text extraction is available.
        if not mime.startswith("image/"):
            response = _responses_create(
                model=MODEL,
                instructions=SYSTEM_WITH_MATERIAL,
                input=prompt + "\n\nSTUDY MATERIAL:\n" + text[:CHUNK_CHARS],
                max_output_tokens=5000,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "study_analysis",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            raw = response.output_text.strip()
        result = json.loads(raw)
    except Exception:
        # Keep the endpoint useful if the model returns non-JSON.
        result = {
            "title": filename,
            "summary": raw,
            "difficulty": "متوسط",
            "understanding": 0,
            "high_yield": [],
            "topics": [],
            "must_memorize": [],
            "must_understand": [],
            "terms": [],
            "comparisons": [],
            "rewrite": raw,
            "study_plan": [],
            "quick_revision": [],
            "questions": [],
            "flashcards": [],
        }

    result["source_file"] = filename
    return result


@app.post("/reset")
def reset():
    last_material.update({
        "name": None,
        "text": "",
        "image_data": None,
        "mime": None,
    })
    return {"ok": True}


# ============================================================
# STUDY AI DEVICE ACTIVATION / DEVELOPER CONTROL
# ============================================================
# This section restores the developer activation API while keeping
# AI chat independent from uploaded files.
#
# Storage is a local JSON file. On Render Free, local disk is not a
# guaranteed permanent database; for a production multi-customer
# system, move this store to a persistent DB. The API is designed so
# that such a migration can be done without changing the Android UI.

from datetime import datetime, timedelta, timezone

ACTIVATION_DB = Path(os.getenv("ACTIVATION_DB_PATH", "/tmp/studyai_licenses.json"))
ADMIN_KEY = os.getenv("STUDYAI_ADMIN_KEY", "")

def _utc_now():
    return datetime.now(timezone.utc)

def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()

def _parse_date(value):
    if not value:
        return None
    value = str(value).strip()
    # Accept YYYY-MM-DD and common Android date formats.
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def _load_licenses():
    try:
        if ACTIVATION_DB.exists():
            data = json.loads(ACTIVATION_DB.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # Preserve the device IDs that were previously activated in this project.
    seeded = {
        "c73ea14226b27b90": {
            "device_id": "c73ea14226b27b90",
            "name": "مستخدم",
            "start_date": "2026-09-16",
            "expiry": "2027-10-10",
            "active": True,
            "status": "active",
            "duration": "manual",
            "created_at": "2026-09-16T00:00:00+00:00",
            "updated_at": "2026-09-16T00:00:00+00:00",
        },
        "1683b276a8ec2aaf": {
            "device_id": "1683b276a8ec2aaf",
            "name": "مستخدم",
            "start_date": "2026-09-16",
            "expiry": "2027-10-10",
            "active": True,
            "status": "active",
            "duration": "manual",
            "created_at": "2026-09-16T00:00:00+00:00",
            "updated_at": "2026-09-16T00:00:00+00:00",
        },
        "8ddb52be1b098611": {
            "device_id": "8ddb52be1b098611",
            "name": "مستخدم",
            "start_date": "2026-09-16",
            "expiry": "2027-10-10",
            "active": True,
            "status": "active",
            "duration": "manual",
            "created_at": "2026-09-16T00:00:00+00:00",
            "updated_at": "2026-09-16T00:00:00+00:00",
        },
    }
    return seeded

licenses = _load_licenses()

def _save_licenses():
    try:
        ACTIVATION_DB.parent.mkdir(parents=True, exist_ok=True)
        tmp = ACTIVATION_DB.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(licenses, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(ACTIVATION_DB)
    except Exception:
        # The service should still answer even when the filesystem is read-only.
        pass

def _normalize_device_id(value):
    return str(value or "").strip().lower()

def _admin_ok(key):
    # Never log the key.
    return bool(ADMIN_KEY) and str(key or "") == ADMIN_KEY

def _license_status(record):
    if not record:
        return {
            "active": False,
            "status": "not_found",
            "message": "الجهاز غير مسجل.",
        }

    expiry = _parse_date(record.get("expiry"))
    now = _utc_now()

    if not record.get("active", False):
        return {
            **record,
            "active": False,
            "status": "revoked",
            "message": "التفعيل ملغى.",
        }

    if expiry and expiry < now:
        record["active"] = False
        record["status"] = "expired"
        return {
            **record,
            "active": False,
            "status": "expired",
            "message": "انتهى الاشتراك.",
        }

    return {
        **record,
        "active": True,
        "status": "active",
        "message": "الجهاز مفعل.",
    }

class LicenseRequest(BaseModel):
    device_id: str = ""
    name: Optional[str] = None
    password: Optional[str] = None
    expiry: Optional[str] = None
    start_date: Optional[str] = None
    duration: Optional[str] = None
    days: Optional[int] = None
    admin_key: Optional[str] = None

def _duration_days(duration, days=None):
    if days is not None:
        try:
            return max(1, int(days))
        except Exception:
            pass

    d = str(duration or "").strip().lower()
    mapping = {
        "7": 7,
        "7 days": 7,
        "7 يوم": 7,
        "14": 14,
        "14 days": 14,
        "14 يوم": 14,
        "1 month": 30,
        "month": 30,
        "1 month/شهر": 30,
        "30": 30,
        "2 months": 60,
        "2 month": 60,
        "60": 60,
    }
    return mapping.get(d)

def _activate_license(req: LicenseRequest, require_admin=True):
    device_id = _normalize_device_id(req.device_id)
    if not device_id:
        raise HTTPException(400, "Device ID مطلوب.")

    if require_admin:
        key = req.admin_key or req.password
        if not _admin_ok(key):
            raise HTTPException(403, "مفتاح المطور غير صحيح.")

    now = _utc_now()
    start = _parse_date(req.start_date) or now

    expiry = _parse_date(req.expiry)
    days = _duration_days(req.duration, req.days)
    if not expiry and days:
        expiry = start + timedelta(days=days)

    if not expiry:
        # Default activation is 30 days when no expiry/duration is supplied.
        expiry = start + timedelta(days=30)

    record = {
        "device_id": device_id,
        "name": (req.name or "مستخدم").strip(),
        "start_date": start.date().isoformat(),
        "expiry": expiry.date().isoformat(),
        "active": True,
        "status": "active",
        "duration": req.duration or (f"{days} days" if days else "manual"),
        "created_at": licenses.get(device_id, {}).get("created_at", _iso(now)),
        "updated_at": _iso(now),
    }
    licenses[device_id] = record
    _save_licenses()

    return {
        "ok": True,
        "success": True,
        "activated": True,
        "active": True,
        "status": "active",
        "device_id": device_id,
        "name": record["name"],
        "start_date": record["start_date"],
        "expiry": record["expiry"],
        "message": "تم تفعيل الجهاز بنجاح.",
    }

# User/device activation endpoint aliases.
@app.post("/activate")
@app.post("/activation")
@app.post("/device/activate")
@app.post("/device/activation")
@app.post("/license/activate")
@app.post("/api/license/activate")
def activate_license(req: LicenseRequest):
    return _activate_license(req, require_admin=True)

@app.get("/activate")
def activate_get(
    device_id: str = "",
    password: str = "",
    name: str = "",
    expiry: str = "",
):
    req = LicenseRequest(
        device_id=device_id,
        password=password,
        name=name,
        expiry=expiry,
    )
    return _activate_license(req, require_admin=True)

# Device status endpoints. These do not expose the admin key.
def _device_status(device_id):
    device_id = _normalize_device_id(device_id)
    if not device_id:
        raise HTTPException(400, "Device ID مطلوب.")

    record = licenses.get(device_id)
    result = _license_status(record)

    if record and result.get("status") == "expired":
        licenses[device_id] = result
        _save_licenses()

    return {
        "ok": True,
        "success": True,
        "device_id": device_id,
        "active": bool(result.get("active", False)),
        "activated": bool(result.get("active", False)),
        "status": result.get("status"),
        "name": result.get("name"),
        "start_date": result.get("start_date"),
        "expiry": result.get("expiry"),
        "message": result.get("message"),
    }

@app.get("/device/status")
@app.get("/device/check")
@app.get("/activation/status")
@app.get("/license/status")
@app.get("/api/license/status")
@app.get("/check-device")
def device_status(device_id: str = ""):
    return _device_status(device_id)

@app.post("/device/status")
@app.post("/device/check")
@app.post("/activation/status")
@app.post("/license/status")
@app.post("/api/license/status")
@app.post("/check-device")
def device_status_post(req: LicenseRequest):
    return _device_status(req.device_id)

# Deactivate/revoke.
@app.post("/deactivate")
@app.post("/device/deactivate")
@app.post("/activation/deactivate")
@app.post("/license/deactivate")
@app.post("/api/license/deactivate")
def deactivate_license(req: LicenseRequest):
    device_id = _normalize_device_id(req.device_id)
    if not device_id:
        raise HTTPException(400, "Device ID مطلوب.")
    if not _admin_ok(req.admin_key or req.password):
        raise HTTPException(403, "مفتاح المطور غير صحيح.")

    if device_id in licenses:
        licenses[device_id]["active"] = False
        licenses[device_id]["status"] = "revoked"
        licenses[device_id]["updated_at"] = _iso(_utc_now())
        _save_licenses()

    return {
        "ok": True,
        "success": True,
        "active": False,
        "activated": False,
        "status": "revoked",
        "device_id": device_id,
        "message": "تم إلغاء التفعيل.",
    }

# Extend/update an existing license.
@app.post("/extend")
@app.post("/device/extend")
@app.post("/activation/extend")
@app.post("/license/extend")
@app.post("/api/license/extend")
def extend_license(req: LicenseRequest):
    device_id = _normalize_device_id(req.device_id)
    if not device_id:
        raise HTTPException(400, "Device ID مطلوب.")
    if not _admin_ok(req.admin_key or req.password):
        raise HTTPException(403, "مفتاح المطور غير صحيح.")

    old = licenses.get(device_id)
    if not old:
        raise HTTPException(404, "Device ID غير موجود.")

    base = _parse_date(old.get("expiry")) or _utc_now()
    days = _duration_days(req.duration, req.days) or 30
    new_expiry = base + timedelta(days=days)

    old["expiry"] = new_expiry.date().isoformat()
    old["active"] = True
    old["status"] = "active"
    old["updated_at"] = _iso(_utc_now())
    if req.name:
        old["name"] = req.name.strip()
    licenses[device_id] = old
    _save_licenses()

    return {
        "ok": True,
        "success": True,
        "active": True,
        "status": "active",
        "device_id": device_id,
        "expiry": old["expiry"],
        "message": "تم تمديد الاشتراك.",
    }

# Developer/admin information.
@app.get("/admin/devices")
@app.get("/admin/licenses")
@app.get("/admin/activations")
def admin_devices(admin_key: str = ""):
    if not _admin_ok(admin_key):
        raise HTTPException(403, "مفتاح المطور غير صحيح.")

    rows = []
    for device_id, record in licenses.items():
        rows.append(_license_status(record))
    return {
        "ok": True,
        "count": len(rows),
        "devices": rows,
        "activations": rows,
    }

@app.post("/admin/devices")
@app.post("/admin/licenses")
@app.post("/admin/activations")
def admin_devices_post(req: LicenseRequest):
    if not _admin_ok(req.admin_key or req.password):
        raise HTTPException(403, "مفتاح المطور غير صحيح.")
    rows = [_license_status(r) for r in licenses.values()]
    return {"ok": True, "count": len(rows), "devices": rows, "activations": rows}

@app.get("/admin/device")
@app.get("/admin/details")
@app.get("/admin/license")
def admin_details(device_id: str = "", admin_key: str = ""):
    if not _admin_ok(admin_key):
        raise HTTPException(403, "مفتاح المطور غير صحيح.")
    return _device_status(device_id)

# Health includes activation subsystem without exposing secrets.
@app.get("/activation/health")
def activation_health():
    return {
        "ok": True,
        "activation_system": True,
        "stored_devices": len(licenses),
        "admin_configured": bool(ADMIN_KEY),
    }
