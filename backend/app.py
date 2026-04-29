import os
import json
import time
import random
import io
import re
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
import csv
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)

app = FastAPI(title="AI vs Real Image Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL_NAME = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")

CSV_FILE = "prediction_results.csv"

SYSTEM_PROMPT = """
You are an expert image authenticity and digital forensics assistant.

Your task is to classify the uploaded image into exactly one of these 3 labels:

AI_GENERATED
REAL_IMAGE
UNCERTAIN

Important:
- You cannot prove an image is real only from visual appearance.
- Do not classify as REAL_IMAGE unless the image has strong natural photographic evidence.
- If the image looks synthetic, overly perfect, digitally rendered, or AI-like, classify as AI_GENERATED.
- If the evidence is weak, mixed, or not enough, classify as UNCERTAIN.
- Never invent metadata, camera details, or source history.
- Judge only from visible image evidence.

Classify as AI_GENERATED when you see signs like:
- overly smooth skin, surfaces, or textures
- unnatural lighting or perfect studio-like lighting
- strange shadows, reflections, or inconsistent light direction
- distorted hands, fingers, teeth, eyes, ears, logos, or text
- unrealistic object edges or blended boundaries
- repeated patterns or duplicated textures
- background that looks too smooth, artificial, or generated
- product-photo look with perfect cleanliness and no real-world imperfections
- objects with wrong geometry, impossible shapes, or inconsistent perspective
- plastic, toy-like, CGI, render, or synthetic appearance

Classify as REAL_IMAGE only when there are strong natural signs like:
- realistic camera noise or compression artifacts
- natural lighting imperfections
- believable shadows and reflections
- real-world messiness, dust, scratches, wrinkles, blur, or uneven texture
- consistent perspective and object geometry
- readable, undistorted text/logos
- realistic background details
- no obvious AI/rendering artifacts

Classify as UNCERTAIN when:
- the image is low quality, cropped, blurry, or lacks detail
- it is a clean product/studio image without enough evidence
- both real and AI signs are present
- there is not enough visual proof to confidently decide

Confidence rules:
- Use 0.85 to 1.0 only when evidence is very strong.
- Use 0.60 to 0.84 when evidence is moderate.
- Use below 0.60 when uncertain.
- For UNCERTAIN, confidence should usually be between 0.30 and 0.60.

Return only valid JSON. No markdown. No explanation outside JSON.

JSON format:
{
  "label": "AI_GENERATED or REAL_IMAGE or UNCERTAIN",
  "confidence": 0.0,
  "reason": "short visual reason for the decision"
}
"""


def clean_json_response(text):
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)

    text = text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_result(result):
    label = str(result.get("label", "UNCERTAIN")).upper().strip()
    if label not in {"AI_GENERATED", "REAL_IMAGE", "UNCERTAIN"}:
        label = "UNCERTAIN"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(confidence, 1.0))

    return {
        "label": label,
        "confidence": confidence,
        "reason": result.get("reason", "No reason provided.")
    }


def save_result_to_csv(filename, result):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                "timestamp", "filename", "label",
                "confidence", "reason", "model_used"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            result.get("label"),
            result.get("confidence"),
            result.get("reason"),
            result.get("model_used")
        ])

def classify_image(image: Image.Image, max_retries=4):
    image = image.convert("RGB")
    image.thumbnail((768, 768))

    models_to_try = [MODEL_NAME, FALLBACK_MODEL_NAME]

    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[SYSTEM_PROMPT, image],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                result = normalize_result(clean_json_response(response.text))
                return {
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "reason": result["reason"],
                    "model_used": model_name
                }

            except ServerError:
                wait = min((2 ** attempt) + random.uniform(0, 1), 10)
                time.sleep(wait)

            except ClientError as e:
                error_text = str(e)
                if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                    reason = "Gemini quota limit reached. Please wait and try again later."
                else:
                    reason = f"Gemini client/API error: {e}"
                return {
                    "label": "UNCERTAIN",
                    "confidence": 0.0,
                    "reason": reason,
                    "model_used": model_name
                }

            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    break
                wait = min((2 ** attempt) + random.uniform(0, 1), 10)
                time.sleep(wait)

    return {
        "label": "UNCERTAIN",
        "confidence": 0.0,
        "reason": "Gemini did not return valid JSON after retries. Please try again.",
        "model_used": f"{MODEL_NAME}, fallback: {FALLBACK_MODEL_NAME}"
    }


@app.get("/")
def home():
    return {
        "message": "AI vs Real Image Detector API is running",
        "model_used": MODEL_NAME
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    result = classify_image(image)
    save_result_to_csv(file.filename, result)
    return {
        "filename": file.filename,
        "result": result,
        "saved_to_csv": True
    }
