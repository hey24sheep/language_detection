# We are using https://huggingface.co/papluca/xlm-roberta-base-language-detection
# License of the model is MIT so we can use it freely

import os
import re

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TextClassificationPipeline)

# from . import MODEL_PATH

MODEL_PATH = "./language_detector/model/roberta_lang_detect"

MODEL_ID = "papluca/xlm-roberta-base-language-detection"

class Model:

    language_lbl = dict({
        'ar': 'Arabic',
        'bg': 'Bulgarian',
        'de': 'German',
        're': 'Modern',
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'hi': 'Hindi',
        'it': 'Italian',
        'ja': 'Japanese',
        'nl': 'Dutch',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'sw': 'Swahili',
        'th': 'Thai',
        'tr': 'Turkish',
        'ur': 'Urdu',
        'vi': 'Vietnamese',
        'zh': 'Chinese',
    })

    pipeline = None

    def __init__(self):
        pass

    def init_pipeline(self):
        self.download_model()
        if self.pipeline is None:
            print('creating pipeline')
            self.pipeline = TextClassificationPipeline(
                model=AutoModelForSequenceClassification.from_pretrained(
                    MODEL_PATH),
                tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH),
            )

    def download_model(self):
        if not os.path.isdir(MODEL_PATH):
            print("model not found!")

            print(f"downloading model {MODEL_ID}")

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

            model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

            tokenizer.save_pretrained(MODEL_PATH)

            model.save_pretrained(MODEL_PATH)

            print(f"model saved to {MODEL_PATH}")


    def supported_languages(self):
        return self.language_lbl

    def detect(self, text: str):
        """Detect language IDs of given texts."""

        self.init_pipeline()

        text = self.clean_text(text)

        if not text:
            return {
                'error':
                'Invalid input, contains symbols, digits or escape characters'
            }

        result = self.pipeline(text)

        label = self.language_lbl[result[0]['label']]

        score = "{:.2f}".format(result[0]['score'])

        return {label: score}

    def clean_text(self, text):
        """Basic cleaning of texts."""

        # remove html markup
        text = re.sub("(<.*?>)", "", text)

        #remove non-ascii and digits
        text = re.sub("(\\W|\\d)", " ", text)

        #remove whitespace
        text = text.strip()
        return text
