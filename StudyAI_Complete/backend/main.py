import os
import base64
import io
import json
import re
import time
from pathlib import Path

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


app = FastAPI(title="Study AI Backend", version="4.0")

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

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
client = OpenAI(api_key=API_KEY) if API_KEY else None

CHUNK_CHARS = int(os.getenv("AI_CHUNK_CHARS", "24000"))
LARGE_FILE_CHARS = int(os.getenv("AI_LARGE_FILE_CHARS", "50000"))
MAX_RETRIES = 4

last_material = {
    "name": None,
    "text": "",
    "image_data": None,
    "mime": None,
}

SYSTEM = """You are Study AI, an academic study assistant specialized in medical and pharmaceutical education.

Core rules:
- Use the uploaded study material as the primary source.
- Do not invent scientific facts, doses, drug names, numbers, units, abbreviations, or exam certainty.
- If the material is unclear or missing the answer, say so.
- If you add knowledge not present in the material, label it exactly as: "معلومة إضافية".
- For Arabic translation, use accurate medical/pharmaceutical terminology and keep the English term beside it when useful.
- Keep the material's meaning; do not silently omit important information.
- Prefer clear Arabic RTL-friendly headings and concise bullets.
- Never claim that a question is guaranteed to appear on an exam.
"""

def require_client():
    if not client:
        raise HTTPException(503, "OPENAI_API_KEY غير مضبوط على الخادم.")


def extract_text(data: bytes, name: str) -> str:
    ext = Path(name).suffix.lower()

    if ext == ".pdf" and PdfReader:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(
            (page.extract_text() or "")
            for page in reader.pages
        )

    if ext == ".docx" and Document:
        doc = Document(io.BytesIO(data))
        return "\n".join(
            p.text for p in doc.paragraphs
            if p.text.strip()
        )

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


def clean_json(raw: str):
    raw = raw.strip()

    if raw.startswith("```"):
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.I
        )
        raw = re.sub(
            r"\s*```$",
            "",
            raw
        )

    start = raw.find("{")
    end = raw.rfind("}")

    if start >= 0 and end > start:
        raw = raw[start:end + 1]

    return json.loads(raw)


def make_input(instruction: str):
    if last_material["image_data"]:
        return [{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": instruction
                },
                {
                    "type": "input_image",
                    "image_url":
                        f"data:{last_material['mime']};base64,"
                        f"{last_material['image_data']}",
                },
            ],
        ]

    return (
        instruction
        + "\n\nSTUDY MATERIAL:\n"
        + last_material["text"]
    )


def _responses_create(**kwargs):
    require_client()

    delay = 2.0

    for attempt in range(MAX_RETRIES):
        try:
            return client.responses.create(**kwargs)

        except Exception as exc:
            msg = str(exc).lower()

            if "rate_limit" not in msg and "429" not in msg:
                raise

            if attempt >= MAX_RETRIES - 1:
                raise

            time.sleep(delay)
            delay *= 2


def ask_ai(
    instruction: str,
    text_override: str | None = None,
    schema=None,
    name="study_analysis",
    max_output_tokens=2500
) -> str:

    require_client()

    if text_override is None:
        input_value = make_input(instruction)
    else:
        input_value = (
            instruction
            + "\n\nSTUDY MATERIAL EXCERPT:\n"
            + text_override
        )

    kwargs = {
        "model": MODEL,
        "instructions": SYSTEM,
        "input": input_value,
        "max_output_tokens": max_output_tokens,
    }

    if schema is not None:
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            }
        }

    response = _responses_create(**kwargs)

    return response.output_text.strip()


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL,
        "material_loaded": bool(last_material["name"])
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    data = await file.read()
    mime = file.content_type or ""
    text = extract_text(data, file.filename)

    if not text and not mime.startswith("image/"):
        raise HTTPException(
            400,
            "لم أستطع استخراج النص من هذا الملف. استخدم PDF/DOCX/PPTX أو صورة واضحة."
        )

    last_material["name"] = file.filename
    last_material["text"] = text
    last_material["mime"] = mime

    last_material["image_data"] = (
        base64.b64encode(data).decode("ascii")
        if mime.startswith("image/")
        else None
    )

    prompt = """حلّل المادة الدراسية تحليلاً شاملاً، ثم أعد النتيجة كـ JSON صالح فقط، بدون Markdown وبدون أي نص خارج JSON.

استخدم هذه المفاتيح حرفياً:

{
  "title": "اسم المادة أو الموضوع",
  "summary": "ملخص واضح",
  "difficulty": "سهل/متوسط/صعب",
  "understanding": 0,
  "high_yield": ["نقاط High-Yield"],
  "topics": ["المواضيع الرئيسية"],
  "must_memorize": ["ما يجب حفظه"],
  "must_understand": ["ما يجب فهمه"],
  "terms": [
    {
      "english": "",
      "arabic": "",
      "explanation": ""
    }
  ],
  "comparisons": ["مقارنات مهمة"],
  "rewrite": "A clear, organized English rewrite of the study material, preserving the original scientific meaning and important terminology.",
  "study_plan": ["خطوات مرتبة لدراسة المادة"],
  "quick_revision": ["نقاط مراجعة سريعة"],
  "questions": [
    {
      "question": "",
      "options": [
        "A) ",
        "B) ",
        "C) ",
        "D) "
      ],
      "answer": "",
      "explanation": ""
    }
  ],
  "flashcards": [
    {
      "front": "",
      "back": ""
    }
  ]
}

اجعل understanding رقماً من 0 إلى 100 يمثل وضوح المادة واستيعاب بنيتها، وليس تشخيصاً لمستوى الطالب.

لا تضع معلومات غير موجودة في المادة إلا إذا وسمتها داخل النص بأنها "معلومة إضافية".
"""

    full_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "difficulty": {"type": "string"},

            "understanding": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100
            },

            "high_yield": {
                "type": "array",
                "items": {"type": "string"}
            },

            "topics": {
                "type": "array",
                "items": {"type": "string"}
            },

            "must_memorize": {
                "type": "array",
                "items": {"type": "string"}
            },

            "must_understand": {
                "type": "array",
                "items": {"type": "string"}
            },

            "terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "english": {"type": "string"},
                        "arabic": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": [
                        "english",
                        "arabic",
                        "explanation"
                    ]
                }
            },

            "comparisons": {
                "type": "array",
                "items": {"type": "string"}
            },

            "rewrite": {
                "type": "string"
            },

            "study_plan": {
                "type": "array",
                "items": {"type": "string"}
            },

            "quick_revision": {
                "type": "array",
                "items": {"type": "string"}
            },

            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "answer": {"type": "string"},
                        "explanation": {"type": "string"}
                   @app.post("/reset")
def reset():

    last_material.update({
        "name": None,
        "text": "",
        "image_data": None,
        "mime": None
    })

    return {
        "ok": True
  }
