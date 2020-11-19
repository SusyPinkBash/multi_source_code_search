from re import finditer
from sys import argv, exit
import pandas as pd
from collections import defaultdict
from gensim import corpora, models, similarities


def handle_camel_case(tokens):
    words = []
    for token in tokens:
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
        words += [m.group(0) for m in matches]
    return words


def filter_stopwords(tokens):
    for token in tokens:
        if token in ['test', 'tests', 'main']:
            return []
    return tokens


def normalize_tokens(tokens):
    return [token.lower() for token in tokens]


def split_underscore(tokens):
    return [word for token in tokens for word in token.split('_')]


def load_csv():
    return pd.read_csv("res/data.csv").fillna(value="")


def create_corpus(df):
    tokens = []
    for _, row in df.iterrows():
        tokens.append(filter_stopwords(normalize_tokens(handle_camel_case(split_underscore(
            [row["name"], row["comment"]])))))

    frequency = defaultdict(int)
    for token in tokens:
        for word in token:
            frequency[word] += 1

    processed = [[token for token in text if frequency[token] > 1] for text in tokens]
    dictionary = corpora.Dictionary(processed)
    bow = [dictionary.doc2bow(text) for text in processed]

    return processed, dictionary, bow


def normalize_query(query):
    return query.strip().lower().split()


def query_tfidf(query, bow, dictionary, df):
    model = models.TfidfModel(bow)
    corpus = model[bow]
    index = similarities.SparseMatrixSimilarity(corpus, num_features=len(dictionary.token2id))

    res = []
    print()
    print('TF-IDF:')
    for i, score in sorted(enumerate(index[model[dictionary.doc2bow(query)]]), key=lambda x: x[1], reverse=True)[:5]:
        print(df.iloc[i], score)
        res.append(df.iloc[i])
    return res


if len(argv) < 2:
    print("Please give as input the query")
    exit(1)

dataframe = load_csv()
processed_corpus, frequencies, bag_of_words = create_corpus(dataframe)
query_to_execute = normalize_query(argv[1])
query_tfidf(query_to_execute, bag_of_words, frequencies, dataframe)
