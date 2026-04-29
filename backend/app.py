import os
import json
import time
import random
import io
import re
import base64
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq, APIError, RateLimitError
import csv
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=API_KEY)

app = FastAPI(title="AI vs Real Image Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

CSV_FILE = "prediction_results.csv"


class ModelResponseError(Exception):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

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
    label = str(result.get("label", "")).upper().strip()
    if label not in {"AI_GENERATED", "REAL_IMAGE", "UNCERTAIN"}:
        raise ValueError("Model returned an unknown label.")

    try:
        confidence = float(result["confidence"])
    except (TypeError, ValueError):
        raise ValueError("Model returned an invalid confidence score.")

    confidence = max(0.0, min(confidence, 1.0))

    return {
        "label": label,
        "confidence": confidence,
        "reason": str(result.get("reason", "")).strip()
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

def image_to_data_url(image: Image.Image):
    image = image.convert("RGB")
    image.thumbnail((768, 768))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_image}"


def classify_image(image: Image.Image, max_retries=4):
    image_data_url = image_to_data_url(image)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url}
                            }
                        ]
                    }
                ],
                temperature=0,
                max_completion_tokens=500,
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content
            result = normalize_result(clean_json_response(response_text))
            return {
                "label": result["label"],
                "confidence": result["confidence"],
                "reason": result["reason"],
                "model_used": MODEL_NAME
            }

        except RateLimitError:
            raise ModelResponseError(
                "Groq rate limit reached. Please wait and try again later.",
                status_code=429
            )

        except APIError as e:
            if attempt == max_retries - 1:
                raise ModelResponseError(f"Groq API error: {e}")
            wait = min((2 ** attempt) + random.uniform(0, 1), 10)
            time.sleep(wait)

        except (json.JSONDecodeError, ValueError):
            if attempt == max_retries - 1:
                break
            wait = min((2 ** attempt) + random.uniform(0, 1), 10)
            time.sleep(wait)

    raise ModelResponseError(
        "Groq did not return a valid classification response after retries."
    )


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
    try:
        result = classify_image(image)
    except ModelResponseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    save_result_to_csv(file.filename, result)
    return {
        "filename": file.filename,
        "result": result,
        "saved_to_csv": True
    }
