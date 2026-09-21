import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from cleaning import (
    clean_text,
    tokenize_and_remove_stopwords,
    stem_tokens,
    lemmatize_tokens,
    correct_spelling,
)


def load_tweets(path, label):
    row = pd.read_csv(path, header=None).iloc[0]
    tweets = row.astype(str).str.strip()
    return pd.DataFrame({"text": tweets.values, "label": label}).reset_index(drop=True)


if __name__ == "__main__":
    df_negative = load_tweets("p00_tweets/processedNegative.csv", "negative")
    df_positive = load_tweets("p00_tweets/processedPositive.csv", "positive")
    df_neutral = load_tweets("p00_tweets/processedNeutral.csv", "neutral")

    df = pd.concat([df_negative, df_positive, df_neutral], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)
    print("after dropping empty rows:", df.shape)

    df["cleaned_text"] = df["text"].apply(clean_text)
    df["tokens"] = df["cleaned_text"].apply(tokenize_and_remove_stopwords)

    X_train_tokens, X_test_tokens, y_train, y_test = train_test_split(
        df["tokens"], df["label"], test_size=0.2, random_state=42
    )

    rows = {
        "tokenization_only": lambda tokens: tokens,
        "stemming": stem_tokens,
        "lemmatization": lemmatize_tokens,
        "misspelling_correction": correct_spelling,
    }

    columns = {
        "binary": CountVectorizer(binary=True),
        "count": CountVectorizer(),
        "tfidf": TfidfVectorizer(),
    }

    algorithms = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "naive_bayes": lambda: MultinomialNB(),
        "svm": lambda: LinearSVC(),
    }

    results = []

    for row_name, row_func in rows.items():
        train_texts = X_train_tokens.apply(row_func).apply(lambda t: " ".join(t))
        test_texts = X_test_tokens.apply(row_func).apply(lambda t: " ".join(t))

        for col_name, vectorizer in columns.items():
            X_train_vec = vectorizer.fit_transform(train_texts)
            X_test_vec = vectorizer.transform(test_texts)

            for algo_name, algo_builder in algorithms.items():
                model = algo_builder()  # fresh, untrained model every time
                model.fit(X_train_vec, y_train)

                y_pred = model.predict(X_test_vec)
                acc = accuracy_score(y_test, y_pred)

                results.append(
                    {
                        "preprocessing": row_name,
                        "vectorizer": col_name,
                        "algorithm": algo_name,
                        "accuracy": round(acc, 4),
                    }
                )
                print(f"{row_name} + {col_name} + {algo_name}: {acc:.4f}")

    results_df = pd.DataFrame(results)
    print("\nTop 10 combinations:")
    print(
        results_df.sort_values("accuracy", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nHow many combinations clear the 0.832 benchmark:")
    print((results_df["accuracy"] >= 0.832).sum(), "out of", len(results_df))
