from language_detector.model import Model

# more test strings from https://huggingface.co/datasets/papluca/language-identification 

model = Model()
langs = model.supported_languages()
print('Supported Languages', langs)
print('Language Count', len(langs.keys()))

result = model.detect("<a><html>!@#")

print(result)

result = model.detect("สำหรับ ><html/> ")

print(result)

result = model.detect("很好,以前从不去评价不知道浪费了多少积分os chefes de defesa da estónia, letónia, lituânia, alemanha, itália, espanha e eslováquia assinarão o acordo para fornecer pessoal e financiamento para o centro.")

print(result)