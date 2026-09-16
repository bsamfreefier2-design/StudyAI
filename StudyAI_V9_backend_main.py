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

origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
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

# Large-file protection: keep each AI request comfortably below the TPM limit.
CHUNK_CHARS = int(os.getenv("AI_CHUNK_CHARS", "24000"))
LARGE_FILE_CHARS = int(os.getenv("AI_LARGE_FILE_CHARS", "50000"))
MAX_RETRIES = 4

# Prototype storage for one user's latest uploaded material.
# For a multi-user production service, replace this with per-user/session storage.
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
- Never claim that a question is guaranteed to appear on an exam. Use "High-Yield" or "مرجّح للمراجعة".
"""

def require_client():
    if not client:
        raise HTTPException(503, "OPENAI_API_KEY غير مضبوط على الخادم.")

def extract_text(data: bytes, name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".pdf" and PdfReader:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
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

def clean_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
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
                {"type": "input_text", "text": instruction},
                {
                    "type": "input_image",
                    "image_url": f"data:{last_material['mime']};base64,{last_material['image_data']}",
                },
            ],
        }]
    return instruction + "\n\nSTUDY MATERIAL:\n" + last_material["text"]

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

def ask_ai(instruction: str, text_override: str | None = None, schema=None, name="study_analysis", max_output_tokens=2500) -> str:
    require_client()
    if text_override is None:
        input_value = make_input(instruction)
    else:
        input_value = instruction + "\n\nSTUDY MATERIAL EXCERPT:\n" + text_override
    kwargs = {
        "model": MODEL,
        "instructions": SYSTEM,
        "input": input_value,
        "max_output_tokens": max_output_tokens,
    }
    if schema is not None:
        kwargs["text"] = {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}
    response = _responses_create(**kwargs)
    return response.output_text.strip()

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "material_loaded": bool(last_material["name"])}

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
        base64.b64encode(data).decode("ascii") if mime.startswith("image/") else None
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
  "terms": [{"english":"", "arabic":"", "explanation":""}],
  "comparisons": ["مقارنات مهمة"],
  "rewrite": "A clear, organized English rewrite of the study material, preserving the original scientific meaning and important terminology.",
  "study_plan": ["خطوات مرتبة لدراسة المادة"],
  "quick_revision": ["نقاط مراجعة سريعة"],
  "questions": [{"question":"", "options":["A) ","B) ","C) ","D) "], "answer":"", "explanation":""}],
  "flashcards": [{"front":"", "back":""}]
}
اجعل understanding رقماً من 0 إلى 100 يمثل وضوح المادة واستيعاب بنيتها، وليس تشخيصاً لمستوى الطالب.
لا تضع معلومات غير موجودة في المادة إلا إذا وسمتها داخل النص بأنها "معلومة إضافية".
"""

    full_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "title": {"type": "string"}, "summary": {"type": "string"}, "difficulty": {"type": "string"},
            "understanding": {"type": "integer", "minimum": 0, "maximum": 100},
            "high_yield": {"type": "array", "items": {"type": "string"}},
            "topics": {"type": "array", "items": {"type": "string"}},
            "must_memorize": {"type": "array", "items": {"type": "string"}},
            "must_understand": {"type": "array", "items": {"type": "string"}},
            "terms": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"english": {"type": "string"}, "arabic": {"type": "string"}, "explanation": {"type": "string"}}, "required": ["english", "arabic", "explanation"]}},
            "comparisons": {"type": "array", "items": {"type": "string"}}, "rewrite": {"type": "string"},
            "study_plan": {"type": "array", "items": {"type": "string"}}, "quick_revision": {"type": "array", "items": {"type": "string"}},
            "questions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}, "answer": {"type": "string"}, "explanation": {"type": "string"}}, "required": ["question", "options", "answer", "explanation"]}},
            "flashcards": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"front": {"type": "string"}, "back": {"type": "string"}}, "required": ["front", "back"]}}
        },
        "required": ["title", "summary", "difficulty", "understanding", "high_yield", "topics", "must_memorize", "must_understand", "terms", "comparisons", "rewrite", "study_plan", "quick_revision", "questions", "flashcards"]
    }

    chunk_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "title": {"type": "string"}, "summary": {"type": "string"},
            "high_yield": {"type": "array", "items": {"type": "string"}},
            "topics": {"type": "array", "items": {"type": "string"}},
            "must_memorize": {"type": "array", "items": {"type": "string"}},
            "must_understand": {"type": "array", "items": {"type": "string"}},
            "terms": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"english": {"type": "string"}, "arabic": {"type": "string"}, "explanation": {"type": "string"}}, "required": ["english", "arabic", "explanation"]}},
            "comparisons": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "summary", "high_yield", "topics", "must_memorize", "must_understand", "terms", "comparisons"]
    }

    # Small/medium files: one request, with a hard input cap that stays below the rate limit.
    if len(text) <= LARGE_FILE_CHARS:
        try:
            raw = ask_ai(prompt, text_override=text[:LARGE_FILE_CHARS], schema=full_schema, name="study_analysis", max_output_tokens=5000)
            result = json.loads(raw)
            return {"filename": file.filename, "model": MODEL, "analysis": result}
        except Exception as exc:
            if "rate_limit" in str(exc).lower() or "429" in str(exc):
                raise HTTPException(429, "تم تجاوز حد التوكنات مؤقتاً. انتظر قليلاً ثم أعد المحاولة.")
            raise HTTPException(502, f"فشل تحليل المادة: {exc}")

    # Large files: analyze sequential chunks, then synthesize the compact chunk results.
    chunks = []
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + CHUNK_CHARS)
        chunks.append(text[pos:end])
        if end >= len(text):
            break
        pos = max(end - 1200, pos + 1)  # small overlap keeps definitions across boundaries

    partials = []
    for i, chunk in enumerate(chunks, 1):
        instruction = f"حلّل الجزء {i} من {len(chunks)} من المادة فقط. استخرج المعلومات المهمة الموجودة فعلياً في هذا الجزء، ولا تخترع معلومات. كن مختصراً لأن هذه نتيجة وسيطة ستُدمج لاحقاً."
        try:
            raw = ask_ai(instruction, text_override=chunk, schema=chunk_schema, name="study_chunk", max_output_tokens=2200)
            partials.append(json.loads(raw))
        except Exception as exc:
            if "rate_limit" in str(exc).lower() or "429" in str(exc):
                raise HTTPException(429, "الملف كبير جداً على حد التوكنات الحالي. أعد المحاولة بعد دقيقة.")
            raise HTTPException(502, f"فشل تحليل الجزء {i}: {exc}")
        time.sleep(1.2)

    combined = json.dumps(partials, ensure_ascii=False)
    final_instruction = """ادمج نتائج تحليل أجزاء المادة التالية في تحليل نهائي واحد. لا تضف معلومات غير موجودة في نتائج الأجزاء. احذف التكرار، وحافظ على أهم النقاط والمصطلحات، ثم أنشئ النتيجة الكاملة بالمخطط المطلوب.\n\nنتائج الأجزاء:\n""" + combined
    try:
        raw = ask_ai(final_instruction, schema=full_schema, name="study_analysis", max_output_tokens=5000)
        result = json.loads(raw)
    except Exception as exc:
        if "rate_limit" in str(exc).lower() or "429" in str(exc):
            raise HTTPException(429, "تم تجاوز حد التوكنات مؤقتاً أثناء تجميع التحليل. أعد المحاولة بعد دقيقة.")
        raise HTTPException(502, f"فشل تجميع التحليل: {exc}")

    return {"filename": file.filename, "model": MODEL, "analysis": result}


FEATURE_PROMPTS = {
    "rewrite": """Rewrite the uploaded study material in clear professional ENGLISH as an organized study handout. Keep the scientific meaning, important details, headings, definitions, mechanisms, numbers, units, and medical/pharmaceutical terminology. Do not translate it to Arabic in this mode. Do not invent information.""",
    "translate_ar": """Translate the uploaded study material into accurate medical/pharmaceutical ARABIC. Keep the original English term beside the Arabic translation when useful. Preserve headings, definitions, mechanisms, numbers, units, and important details. Do not invent information.""",
    "study": """أنشئ خطة دراسة عملية لهذه المادة: ترتيب المواضيع، ماذا أحفظ، ماذا أفهم، ثم أسئلة ومراجعة. اجعلها مناسبة لطالب طب/صيدلة.""",
    "questions": """أنشئ أسئلة امتحانية من المادة، متنوعة بين MCQ وأسئلة فهم. اذكر الإجابة الصحيحة وسببها. لا تدّعِ أن أي سؤال مضمون في الامتحان.""",
    "flashcards": """أنشئ Flashcards تغطي التعاريف والمصطلحات والآليات والأرقام والمقارنات المهمة في المادة. كل بطاقة سؤال/مصطلح ثم جواب مختصر.""",
    "terms": """استخرج أهم المصطلحات الطبية والصيدلانية من المادة في جدول نصي: English | العربية | شرح مبسط.""",
    "revision": """أنشئ مراجعة سريعة جداً للمادة: أهم ما يجب حفظه، أهم ما يجب فهمه، الأخطاء الشائعة، ونقاط High-Yield.""",
    "progress": """قيّم بنية المادة وصعوبتها ووضوحها، ثم أعطِ خطة لتحسين الدراسة. لا تدّعي معرفة مستوى الطالب الحقيقي دون بيانات اختبار.""",
    "drug": """اعمل بطاقة تعليمية عن الدواء أو المادة الفعالة المذكورة في السؤال اعتماداً على المادة المرفوعة أولاً. اذكر: الاسم العلمي، الفئة الدوائية، آلية العمل إذا كانت موجودة، الاستعمالات العامة المذكورة، الأشكال الصيدلانية/التراكيز إن وردت، أهم التحذيرات أو الآثار الجانبية إن وردت. لا تعطِ جرعة شخصية أو وصفة علاجية. إذا لم تكن المعلومة في الملف فقل ذلك بوضوح، وأي معرفة إضافية ابدأها بعبارة "معلومة إضافية".""",
}


class FeatureRequest(BaseModel):
    feature: str
    query: str = ""

@app.post("/feature")
def feature(body: FeatureRequest):
    if not last_material["name"]:
        raise HTTPException(400, "ارفع المادة أولاً.")
    instruction = FEATURE_PROMPTS.get(body.feature)
    if not instruction:
        raise HTTPException(400, "الميزة غير معروفة.")
    if body.feature == "drug":
        if not body.query.strip():
            raise HTTPException(400, "اكتب اسم الدواء أو المادة الفعالة.")
        instruction += "\n\nاسم الدواء/المادة الفعالة المطلوب تحليلها: " + body.query.strip()
    return {"feature": body.feature, "answer": ask_ai(instruction)}

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(body: ChatRequest):
    if not last_material["name"]:
        return {"answer": "ارفع مادة أولاً حتى أجيب بناءً عليها."}
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "اكتب السؤال.")
    instruction = f"""أجب عن سؤال الطالب اعتماداً على المادة المرفوعة أولاً.
إذا لم تكن الإجابة موجودة في المادة، قل ذلك بوضوح. وإذا استخدمت معرفة إضافية، ابدأ فقرتها بـ "معلومة إضافية".

سؤال الطالب:
{q}
"""
    return {"answer": ask_ai(instruction)}

@app.post("/reset")
def reset():
    last_material.update({"name": None, "text": "", "image_data": None, "mime": None})
    return {"ok": True}
