import pandas as pd
from lingua import Language,LanguageDetectorBuilder
import spacy
nlp = spacy.load("en_core_web_sm")

from transformers import pipeline
llm_sentiment = pipeline("sentiment-analysis")

#from flores import code_mapping
#code_mapping = dict(sorted(code_mapping.items(), key=lambda item: item[0]))
#translator = pipeline("translation",model="facebook/nllb-200-distilled-600M",
#    tokenizer="facebook/nllb-200-distilled-600M")
translator = pipeline("translation",model="facebook/nllb-200-distilled-1.3B",
    tokenizer="facebook/nllb-200-distilled-1.3B")


from textblob import TextBlob
# build detector for all Language
detector = LanguageDetectorBuilder.from_all_languages().build()

REMOVED_LANGUAGES = {"la"}
LANG_MAP = {
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "hi": "hin_Deva",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "zh-cn": "zho_Hans",
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "th": "tha_Thai",
    "cs": "ces_Latn",
    "sq": "als_Latn",
    "id": "ind_Latn",
    "ro": "ron_Latn",
    "tr": "tur_Latn",
    "yo": "yor_Latn",
    "zh": "zho_Hans",
    "pt": "por_Latn",
    "tn": "tsn_Latn",
    "cy": "cym_Latn",
    "lt": "lit_Latn",
}

def translateNLLB(text: str,src_lang: str, tgt_lang: str):
    if src_lang in REMOVED_LANGUAGES:
        return text
    src_code = LANG_MAP[src_lang]
    tgt_code = LANG_MAP[tgt_lang]
    print(src_code)
    print(tgt_code)
    trans = translator(text,src_lang = src_code, tgt_lang = tgt_code)
    return trans[0]["translation_text"]

def isAboutAdobe(text: str):
    has_adobe = False
    if "adobe" in text.lower():
        has_adobe = True
    if "bridge" in text.lower():
        has_adobe = True
    return has_adobe

# Read the local CSV file
df = pd.read_csv("./tweets.csv")
#print(df.head())
print(df.info())


results = []
positive = []
negative = []
others = []

for row in df.itertuples():
    text = row.Tweet_Content
    lang = detector.detect_language_of(text)
    print(lang)
    print(lang.name)
    print(lang.iso_code_639_1)
    ln_code = lang.iso_code_639_1.name
    aboutAdobe = isAboutAdobe(text)
    if ln_code != "EN":
        print("ORG-TEXT:" + text)
        text = translateNLLB(text,ln_code.lower(),"en") 
    print(text)
    #print(ln_code)
    #if not isAboutAdobe(text):
    #    continue
    blob = TextBlob(text)
    noun_phrases = blob.noun_phrases
    #print(noun_phrases)
    print(f"Polarity: {blob.sentiment.polarity}")      # -1 (negative) to 1 (positive)
    print(f"Subjectivity: {blob.sentiment.subjectivity}")  # 0 (objective) to 1 (subjective)
    sentiment = llm_sentiment(text)
    print(sentiment)
    doc = nlp(text)
    #companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    #print(companies)
    print("---------")
    # Get all original columns as dict
    row_dict = row._asdict()
    # Add new columns
    row_dict['translated_text'] = text 
    if aboutAdobe:
        if sentiment[0]["label"] == "POSITIVE":
            positive.append(row_dict)
        else:
            negative.append(row_dict)
    else:
        others.append(row_dict)
    #row_dict['is_about_adobe'] = aboutAdobe 
    #results.append(row_dict)

# Create new DataFrame with all columns
#result_df = pd.DataFrame(results)
positive_df = pd.DataFrame(positive)
negative_df= pd.DataFrame(negative)
others_df = pd.DataFrame(others)




# Save to CSV
#result_df.to_csv('processed_tweets.csv', index=False)
positive_df.to_csv('positive_tweets.csv', index=False)
negative_df.to_csv('negative_tweets.csv', index=False)
others_df.to_csv('others_tweets.csv', index=False)


#print(f"Processed {len(results)} rows")
#print(result_df.head())
