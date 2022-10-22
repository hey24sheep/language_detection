from fastapi import FastAPI

from language_detector.model import Model

from pydantic import BaseModel


class DetectInputModel(BaseModel):
    text: str


app = FastAPI()

model = Model()


@app.get("/")
def read_root():
    return {"status": "working", "ver": "v1"}


@app.get("/langs")
def read_root():
    langs = model.supported_languages()
    return {"languages": langs, "count": f"{len(langs.keys())}"}


@app.post("/lang_id/")
def read_item(input: DetectInputModel):
    return model.detect(input.text)
