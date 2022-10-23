# We are using https://huggingface.co/papluca/xlm-roberta-base-language-detection
# License of the model is MIT so we can use it freely

import os
import re
from langdetect import detect_langs
from iso639 import languages

from langdetect import DetectorFactory


class Model:
    language_lbl = [
        "af", "ar", "bg", "bn", "ca", "cs", "cy", "da", "de", "el", "en", "es",
        "et", "fa", "fi", "fr", "gu", "he", "hi", "hr", "hu", "id", "it", "ja",
        "kn", "ko", "lt", "lv", "mk", "ml", "mr", "ne", "nl", "no", "pa", "pl",
        "pt", "ro", "ru", "sk", "sl", "so", "sq", "sv", "sw", "ta", "te", "th",
        "tl", "tr", "uk", "ur", "vi", "zh-cn", "zh-tw"
    ]

    lang_keys = []

    def __init__(self):
        DetectorFactory.seed = 0
        pass


    def supported_languages(self):
        if len(self.lang_keys) > 0:
            return self.lang_keys

        for l in self.language_lbl:
            if l == "zh-cn":
                self.lang_keys.append('Chinese')
            elif l == "zh-tw":
                self.lang_keys.append('Taiwanese Mandarin')
            else:
                self.lang_keys.append(languages.get(alpha2=l).name)

        return self.lang_keys


    def detect(self, text: str):
        """Detect language IDs of given texts."""

        text = self.clean_text(text)

        if not text:
            return {
                'error':
                'Invalid input, contains symbols, digits or escape characters'
            }

        detect_results = detect_langs(text)[0]

        if detect_results.lang == "zh-cn":
            label = "Chinese"
        elif detect_results.lang == "zh-tw":
            label = "Taiwanese Mandarin"
        else:
            label = languages.get(alpha2=detect_results.lang).name

        score = "{:.2f}".format(detect_results.prob)

        return {label: score}

    def clean_text(self, text):
        """Basic cleaning of texts."""

        # remove html markup
        text = re.sub("(<.*?>)", "", text)

        # remove non-ascii and digits
        text = re.sub("(\\W|\\d)", "", text)

        # remove whitespace
        text = text.strip()

        return text
