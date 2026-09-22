# tweets-git

Sentiment classification on tweets, built as a 42/1337 school project. The script
loads a small labeled tweet dataset, runs it through several text-preprocessing
and vectorization strategies, and compares classifier accuracy across all
resulting combinations to find the best pipeline.

## Dataset

`p00_tweets/` contains three CSV files, one label per file:

- `processedPositive.csv`
- `processedNegative.csv`
- `processedNeutral.csv`

Each file is a single row of comma-separated tweet texts (no header). `main.py`
loads all three, assigns the corresponding label, and concatenates them into one
shuffled dataset.

## Pipeline

For every tweet, `cleaning.py` provides the preprocessing building blocks:

- `clean_text` — lowercases, strips URLs, @mentions, `#`, and non-letter
  characters
- `tokenize_and_remove_stopwords` — tokenizes with NLTK and removes English
  stopwords
- `stem_tokens` — Porter stemming
- `lemmatize_tokens` — WordNet lemmatization
- `correct_spelling` — spelling correction via `pyspellchecker`

`main.py` then does a grid search over:

- **Preprocessing**: tokenization only, stemming, lemmatization, misspelling
  correction
- **Vectorizer**: binary `CountVectorizer`, count `CountVectorizer`, `TfidfVectorizer`
- **Algorithm**: Logistic Regression, Multinomial Naive Bayes, Linear SVM

for a total of 4 × 3 × 3 = 36 combinations, each trained/evaluated on an
80/20 train-test split. It prints the accuracy of every combination, the
top 10 by accuracy, and how many clear a 0.832 benchmark.

## Setup

```bash
pip install -r requirements.txt
```

NLTK data (stopwords, punkt, wordnet) is downloaded automatically the first
time `cleaning.py` is imported.

## Usage

```bash
python main.py
```
