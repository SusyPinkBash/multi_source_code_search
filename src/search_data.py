from re import finditer
from sys import argv, exit
import pandas as pd
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel, LsiModel, Doc2Vec
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity



def start(query):
    dataframe = load_csv("res/data.csv")
    results_dictionary = compute_results(query, dataframe)
    results = pd.DataFrame(data=print_queries(results_dictionary, dataframe),
                           columns=['name', "file", "line", "type", "comment", "search"])
    results.to_csv('res/search_data.csv', index=False, encoding='utf-8')


def compute_results(query, dataframe):
    processed_corpus, frequencies, bag_of_words = create_corpus(dataframe)
    query_to_execute = normalize_query(query)
    results_dictionary = {
        "FREQ": filter_results(query_frequency(query_to_execute, bag_of_words, frequencies)),
        "TF-IDF": filter_results(query_tfidf(query_to_execute, bag_of_words, frequencies)),
        "LSI": filter_results(query_lsi(query_to_execute, bag_of_words, frequencies)),
        "Doc2Vec": query_doc2vec(query_to_execute, processed_corpus)
    }
    return results_dictionary


def load_csv(path):
    return pd.read_csv(path).fillna(value="")


def create_corpus(df):
    tokens = [filter_stopwords(normalize_tokens(handle_camel_case(split_underscore(
        [row["name"]] + split_space(row["comment"]))))) for _, row in df.iterrows()]

    frequency = defaultdict(int)
    for token in tokens:
        for word in token:
            frequency[word] += 1

    processed = [[token for token in text if frequency[token] > 1] for text in tokens]
    dictionary = Dictionary(processed)
    bow = [dictionary.doc2bow(text) for text in processed]

    return processed, dictionary, bow


def split_space(text):
    return text.split(' ')


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


def filter_results(arrg):
    return [i for i, v in sorted(enumerate(arrg), key=lambda x: x[1], reverse=True)[:5]]


def query_doc2vec(query, corpus):
    model = get_doc2vec_model(get_doc2vec_read_corpus(corpus))
    return [index for (index, _) in model.docvecs.most_similar([model.infer_vector(query)], topn=5)]


def get_doc2vec_read_corpus(corpus):
    return [TaggedDocument(simple_preprocess(' '.join(element)), [index])
            for index, element in enumerate(corpus)]


def get_doc2vec_model(corpus):
    model = Doc2Vec(vector_size=300, min_count=2, epochs=77)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def create_result_dataframe(queries_dictionary, df):
    for key, values in queries_dictionary.items():
        for index in sorted(values):
            row = df.iloc[index]
            yield [row["name"], row["file"], row["line"], row["type"], row["comment"], key]


def print_queries(queries_dictionary, df):
    for key, values in queries_dictionary.items():
        print(key)
        for index in sorted(values):
            row = df.iloc[index]
            print("document:", index)
            print(row, '\n')
            yield [row["name"], row["file"], row["line"], row["type"], row["comment"], key]
        print()


if len(argv) < 2:
    print("Please give as input the query")
    exit(1)

start(argv[1])
