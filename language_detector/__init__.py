# We are using https://huggingface.co/papluca/xlm-roberta-base-language-detection
# License of the model is MIT so we can use it freely

import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "./language_detector/model/roberta_lang_detect"

MODEL_ID = "papluca/xlm-roberta-base-language-detection"

if not os.path.isdir(MODEL_PATH):
    print("model not found!")

    print(f"downloading model {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    tokenizer.save_pretrained(MODEL_PATH)

    model.save_pretrained(MODEL_PATH)

    print(f"model saved to {MODEL_PATH}")
