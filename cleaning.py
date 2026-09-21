import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer, WordNetLemmatizer
from spellchecker import SpellChecker

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


nltk.download("wordnet")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()


def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]


def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]


def correct_spelling(tokens):
    corrected = []
    for word in tokens:
        fix = spell.correction(word)
        corrected.append(fix if fix else word)  # keep original if no correction found
    return corrected
