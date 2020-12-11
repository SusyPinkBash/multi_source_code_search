from datetime import datetime
import string
from os import path
import pandas as pd
import pickle as pkl
from re import finditer
from sys import argv, exit
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel, LsiModel, Doc2Vec
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity


def start(query):
    dataframe = pd.read_csv("res/data.csv").fillna(value="")
    results_dictionary, _ = compute_results(query, dataframe)
    results = pd.DataFrame(data=create_result_dataframe(results_dictionary, dataframe),
                           columns=['name', "file", "line", "type", "comment", "search"])
    pd.options.display.max_colwidth = 200
    print_results(results)
    results.to_latex('res/search_data.tex', index=False, encoding='utf-8')
    results.to_csv('res/search_data.csv', index=False, encoding='utf-8')


def compute_results(query, dataframe):
    processed_corpus, frequencies, bag_of_words = get_data(dataframe)
    query_to_execute = normalize_query(query)
    results = {
        "FREQ": query_frequency(query_to_execute, bag_of_words, frequencies),
        "TF-IDF": query_tfidf(query_to_execute, bag_of_words, frequencies)
    }
    vectors = dict()
    results["LSI"], vectors["LSI"] = query_lsi(query_to_execute, bag_of_words, frequencies)
    results["Doc2Vec"], vectors["Doc2Vec"] = query_doc2vec(query_to_execute, processed_corpus)
    return results, vectors


def get_data(df):
    return load_data_files() if exists_data_files() else create_data(df)


def create_data(df):
    tokens = [filter_stopwords(normalize_tokens(handle_camel_case(split_underscore(
        [row["name"]] + split_space(row["comment"]))))) for _, row in df.iterrows()]

    frequency = defaultdict(int)
    for token in tokens:
        for word in token:
            frequency[word] += 1

    corpus = [[token for token in text if frequency[token] > 1] for text in tokens]
    dictionary = Dictionary(corpus)
    bow = [dictionary.doc2bow(text) for text in corpus]

    save_data(corpus, 'corpus')
    save_data(dictionary, 'dictionary')
    save_data(bow, 'bow')
    return corpus, dictionary, bow


def exists_data_files():
    return exists_file('corpus') and exists_file('dictionary') and exists_file('bow')


def exists_file(name):
    return path.exists('res/' + name + '.pkl')


def load_data_files():
    return load_file('corpus'), load_file('dictionary'), load_file('bow')


def save_data(data, name):
    pkl.dump(data, open('res/' + name + '.pkl', "wb"), protocol=pkl.HIGHEST_PROTOCOL)


def load_file(name):
    return pkl.load(open('res/' + name + '.pkl', "rb"))


def split_space(text):
    return text.translate(str.maketrans('', '', string.punctuation)).split(' ') if text != "" else []


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


def save_model(model, name):
    save_data(model, 'model_' + name)


def query_frequency(query, bow, dictionary):
    return filter_results(get_freq_model(bow, dictionary)[dictionary.doc2bow(query)])


def get_freq_model(bow, dictionary):
    return load_file('model_freq') if exists_file('model_freq') else create_freq_model(bow, dictionary)


def create_freq_model(bow, dictionary):
    model = SparseMatrixSimilarity(bow, num_features=len(dictionary.token2id))
    save_model(model, 'freq')
    return model


def query_tfidf(query, bow, dictionary):
    model = get_tfidf_model(bow)
    matrix = get_tfidf_matrix(model, bow, dictionary)
    return filter_results(matrix[model[dictionary.doc2bow(query)]])


def get_tfidf_model(bow):
    return load_file('model_tfidf') if exists_file('model_tfidf') else create_tfidf_model(bow)


def create_tfidf_model(bow):
    model = TfidfModel(bow)
    save_model(model, 'tfidf')
    return model


def get_tfidf_matrix(model, bow, dictionary):
    return load_file('matrix_tfidf') if exists_file('matrix_tfidf') else create_tfidf_matrix(model, bow, dictionary)


def create_tfidf_matrix(model, bow, dictionary):
    matrix = SparseMatrixSimilarity(model[bow], num_features=len(dictionary.token2id))
    save_data(matrix, 'matrix_tfidf')
    return model


def query_lsi(query, bow, dictionary):
    model = get_lsi_model(bow, dictionary)
    vector = model[dictionary.doc2bow(query)]
    result = abs(MatrixSimilarity(model[bow])[vector])
    embedding = [[value for _, value in vector]] + [[value for _, value in model[bow][i]] for i, value in
                                                    sorted(enumerate(result), key=lambda x: x[1], reverse=True)[:5]]
    return filter_results(result), embedding


def get_lsi_model(bow, dictionary):
    return load_file('model_lsi') if exists_file('model_lsi') else create_lsi_model(bow, dictionary)


def create_lsi_model(bow, dictionary):
    model = LsiModel(bow, id2word=dictionary, num_topics=300)
    save_model(model, 'lsi')
    return model


def filter_results(arrg):
    return [i for i, v in sorted(enumerate(arrg), key=lambda x: x[1], reverse=True)[:5]]


def query_doc2vec(query, corpus):
    model = get_doc2vec_model(get_doc2vec_corpus(corpus))
    vector = model.infer_vector(query)
    similar = model.docvecs.most_similar([vector], topn=5)
    return [index for (index, _) in similar], \
           [list(vector)] + [list(model.infer_vector(corpus[index])) for index, _ in similar]


def get_doc2vec_corpus(corpus):
    return [TaggedDocument(simple_preprocess(' '.join(element)), [index])
            for index, element in enumerate(corpus)]


def get_doc2vec_model(corpus):
    return load_file('model_doc2vec') if exists_file('model_doc2vec') else create_doc2vec_model(corpus)


def create_doc2vec_model(corpus):
    model = Doc2Vec(vector_size=300, min_count=2, epochs=77)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    save_model(model, 'doc2vec')
    return model


def create_result_dataframe(queries_dictionary, df):
    for key, values in queries_dictionary.items():
        for index in sorted(values):
            row = df.iloc[index]
            yield [row["name"], row["file"], row["line"], row["type"], row["comment"], key]


def print_results(df):
    grouped = df.groupby(['search'])
    for key, item in grouped:
        print(grouped.get_group(key), "\n\n")


if len(argv) < 2:
    print("Please give as input the query")
    exit(1)

start(argv[1])
