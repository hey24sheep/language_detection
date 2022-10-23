from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from language_detector.model import Model


class DetectInputModel(BaseModel):
    text: str


app = FastAPI(
    title="Language Detection",
    description=
    "Opening a simple rest API endpoint to detect language from the given text",
    version="1.0.0",
    contact={
        "name": "Heyramb Narayan Goyal",
        "url": "https://hey24sheep.com",
        "email": "hey24sheep@gmail.com",
    },
    license_info={
        "name":
        "MIT",
        "url":
        "https://raw.githubusercontent.com/hey24sheep/language_detection/main/LICENSE",
    },
)


model = Model()


@app.get("/")
def health():
    return {"status": "working", "ver": "v1"}


@app.get("/langs")
def get_supported_languages():
    langs = model.supported_languages()
    return {"languages": langs, "count": f"{len(langs)}"}


@app.post("/lang_id/")
def detect_language(input: DetectInputModel):
    result = model.detect(input.text)
    if 'error' in result:
        raise HTTPException(status_code=400, 
        detail=result['error'])
    return result
