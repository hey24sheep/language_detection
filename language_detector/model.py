import re

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TextClassificationPipeline)

from . import MODEL_PATH


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
        if self.pipeline is None:
            print('creating pipeline')
            self.pipeline = TextClassificationPipeline(
                model=AutoModelForSequenceClassification.from_pretrained(
                    MODEL_PATH),
                tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH),
            )

    def supported_languages(self):
        return self.language_lbl

    def detect(self, text: str):
        """Detect language IDs of given texts."""

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
