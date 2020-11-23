from re import finditer
from sys import argv, exit
import pandas as pd
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel, doc2vec
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity


def start(query):
    dataframe = load_csv()
    processed_corpus, frequencies, bag_of_words = create_corpus(dataframe)
    query_to_execute = normalize_query(argv[1])
    results_dictionary = {
        "FREQ": query_frequency(query_to_execute, bag_of_words, frequencies),
        "TF-IDF": query_tfidf(query_to_execute, bag_of_words, frequencies),
        "LSI": query_lsi(query_to_execute, bag_of_words, frequencies),
        # "Doc2Vec": query_doc2vec(query_to_execute, processed_corpus)
    }
    data = print_queries(results_dictionary, dataframe)
    results = pd.DataFrame(data=data, columns=['name', "file", "line", "type", "comment", "search"])
    results.to_csv('res/search_data.csv', index=False, encoding='utf-8')


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
    dictionary = Dictionary(processed)
    bow = [dictionary.doc2bow(text) for text in processed]

    return processed, dictionary, bow


def split_underscore(tokens):
    return [word for token in tokens for word in token.split('_')]


def handle_camel_case(tokens):
    words = []
    for token in tokens:
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
        words += [m.group(0) for m in matches]
    return words


def normalize_tokens(tokens):
    return [token.lower() for token in tokens]


def filter_stopwords(tokens):
    for token in tokens:
        if token in ['test', 'tests', 'main']:
            return []
    return tokens


def normalize_query(query):
    return query.strip().lower().split()


def query_frequency(query, bow, dictionary):
    return SparseMatrixSimilarity(bow, num_features=len(dictionary.token2id))[dictionary.doc2bow(query)]


def query_tfidf(query, bow, dictionary):
    model = TfidfModel(bow)
    return SparseMatrixSimilarity(model[bow], num_features=len(dictionary.token2id))[model[dictionary.doc2bow(query)]]


def query_lsi(query, bow, dictionary):
    model = LsiModel(bow, id2word=dictionary, num_topics=300)
    return abs(MatrixSimilarity(model[bow])[model[dictionary.doc2bow(query)]])


def query_doc2vec(query, corpus):
    return "TODO"


def print_queries(queries_dictionary, df):
    results = []
    for key, values in queries_dictionary.items():
        print(key)
        for index, value in sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:5]:
            print("document:", index)
            row = df.iloc[index]
            results.append([row["name"], row["file"], row["line"], row["type"], row["comment"], key])
            print(row, value, '\n')
    print()
    return results


if len(argv) < 2:
    print("Please give as input the query")
    exit(1)


start(argv[1])

