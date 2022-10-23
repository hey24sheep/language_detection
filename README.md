# Language Detection
Language detection using  [langdetect](https://github.com/Mimino666/langdetect) and FastAPI as the backend. It supports 55 languages. As this is a n-gram based detector.

### **NOTE**

I have also used `RoBERTa` based detection project as well, you can find that project here [GitLab Project](https://gitlab.com/hey24sheep/language_detection). It is a huge project, almost 2GB. Also, it only supports 20 languages. But it is a lot more reliable.

# Setup
- Install all dependencies, run `pip install -r requirements.txt` or `install.sh` 
  - OR, 
    - Create `python -m venv .env` virtual env (in case you already have an 'env', make sure to delete it)
    - run `activate_env_linux.sh`(if using linux) or `activate_env_windows.bat`
    - then install all the dependencies
- Run `run_server.sh` to consume/test rest endpoints

# Validations
## Input 
There are 3 input validations

- Remove all `html markup`
- Remove all `non-ascii and digits`
- Remove all `whitespaces`
  

# Tests
## Model Unit Tests
- Run `python local_detector_test.py` to test the model with some test data

## API Unit Tests
- Run `pytest` or `run_tests.sh` to test the rest API endpoints

## API Docs & Specifications
 All docs & specs are automatically generated by Swagger or ReDoc built-in within FastAPI.

API Docs / Live Test
 - To check `Swagger` use `/docs`
 - To check `Redoc` use `/redoc`
  
API Specification
- Autogenerates `OpenAPI` based spec file
  - Can be checked at `/openapi.json`

# Detection / Identification

## Detection & Supported Languages
I am using this awesome library [langdetect](https://github.com/Mimino666/langdetect). It is a port of original library of the same name by Google.

As of now used the library supports the following **55** languages.

```
'Afrikaans', 'Arabic', 'Bulgarian', 'Bengali', 'Catalan', 'Czech', 'Welsh', 'Danish', 'German', 'Modern Greek (1453-)', 'English', 'Spanish', 'Estonian', 'Persian', 'Finnish', 'French', 'Gujarati', 'Hebrew', 'Hindi', 'Croatian', 'Hungarian', 'Indonesian', 'Italian', 'Japanese', 'Kannada', 'Korean', 'Lithuanian', 'Latvian', 'Macedonian', 'Malayalam', 'Marathi', 'Nepali (macrolanguage)', 'Dutch', 'Norwegian', 'Panjabi', 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Slovak', 'Slovenian', 'Somali', 'Albanian', 'Swedish', 'Swahili (macrolanguage)', 'Tamil', 'Telugu', 'Thai', 'Tagalog', 'Turkish', 'Ukrainian', 'Urdu', 'Vietnamese', 'Chinese', 'Taiwanese Mandarin'
```

## How the library works?
You can read this presentation by [Google from 2010](https://www.slideshare.net/shuyo/language-detection-library-for-java)

## Can we use something newer?
Yes, we can use [xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection?text=I+like+you.+I+love+you`) model from HuggingFace. Which is a fine tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) trained on [Language ID Dataset](https://huggingface.co/datasets/papluca/language-identification#additional-information).

## Why haven't I used it?
I have used it, you can find that project here [GitLab Project](https://gitlab.com/hey24sheep/language_detection). But, it cannot be deployed to free resources due to its huge size.

It's a big +1.5GB model, even dependecies are huge too. Second, it only supports 20 languages.

# Flow

- User makes `POST` call to `/lang_id`
  - "Json"  input
  - "text" key with your input text
    - `{"text": "Le parole est l\u0027ombre du fait"}`
  
- Input is validated (check section `Validation` for more details)
  - If `empty` or `null`, throw `BadRequest`
  - else, return `detection results`

## Author
[Hey24sheep](https://hey24sheep.com)


