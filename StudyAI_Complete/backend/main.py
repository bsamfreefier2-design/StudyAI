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
