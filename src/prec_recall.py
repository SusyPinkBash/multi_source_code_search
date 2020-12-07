import itertools
from datetime import datetime

import string
import pandas as pd
from os import path
import pickle as pkl
import seaborn as sns
from re import finditer
from sys import argv, exit
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel, LsiModel, Doc2Vec
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity

##################
def get_results(query, dataframe):
    results_dictionary, vectors = compute_results(query, dataframe)
    return pd.DataFrame(data=create_result_dataframe(results_dictionary, dataframe),
                        columns=['name', "file", "line", "type", "comment", "search"]), vectors


def compute_results(query, dataframe):
    processed_corpus, frequencies, bag_of_words = create_data(dataframe)
    query_to_execute = normalize_query(query)
    results = {
        "FREQ": filter_results(query_frequency(query_to_execute, bag_of_words, frequencies)),
        "TF-IDF": filter_results(query_tfidf(query_to_execute, bag_of_words, frequencies))
    }
    vectors = dict()
    results["LSI"], vectors["LSI"] = query_lsi(query_to_execute, bag_of_words, frequencies)
    results["Doc2Vec"], vectors["Doc2Vec"] = query_doc2vec(query_to_execute, processed_corpus)
    return results, vectors


def create_data(df):
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


def query_frequency(query, bow, dictionary):
    return SparseMatrixSimilarity(bow, num_features=len(dictionary.token2id))[dictionary.doc2bow(query)]


def query_tfidf(query, bow, dictionary):
    model = TfidfModel(bow)
    return SparseMatrixSimilarity(model[bow], num_features=len(dictionary.token2id))[model[dictionary.doc2bow(query)]]


def query_lsi(query, bow, dictionary):
    model = LsiModel(bow, id2word=dictionary, num_topics=300)
    vector = model[dictionary.doc2bow(query)]
    result = abs(MatrixSimilarity(model[bow])[vector])
    embedding = [[value for _, value in vector]] + [[value for _, value in model[bow][i]] for i, value in
                                                    sorted(enumerate(result), key=lambda x: x[1], reverse=True)[:5]]
    return filter_results(result), embedding


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
    return pkl.load(open('res/doc2vec.pkl', "rb")) if path.exists('res/doc2vec.pkl') else create_doc2vec_model(corpus)


def create_doc2vec_model(corpus):
    model = Doc2Vec(vector_size=300, min_count=2, epochs=77)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    pkl.dump(model, open('res/doc2vec.pkl', "wb"), protocol=pkl.HIGHEST_PROTOCOL)
    return model


def create_result_dataframe(queries_dictionary, df):
    for key, values in queries_dictionary.items():
        for index in sorted(values):
            row = df.iloc[index]
            yield [row["name"], row["file"], row["line"], row["type"], row["comment"], key]


#####################################

class Truth:
    def __init__(self, query, name, path):
        self.name = name
        self.path = path
        self.query = query.lower()


class Stat:
    def __init__(self, precisions, recalls):
        self.precisions = precisions
        self.recalls = recalls


def start(path_ground_truth):
    dataframe = pd.read_csv("res/data.csv").fillna(value="")
    ground_truth, queries = parse_ground_truth(path_ground_truth)
    scores, vectors = compute_precision_recall(ground_truth, dataframe)
    plot_vectors(compute_tsne(vectors), queries)
    print_scores(scores)


def parse_ground_truth(path_ground_truth):
    classes, queries = [], []
    for entry in open(path_ground_truth, "r").read().split("\n\n"):
        data = entry.split("\n")
        classes.append(Truth(data[0], data[1], data[2]))
        queries.append(data[0])
    return classes, queries


def compute_precision_recall(ground_truth, dataframe):
    scores = {"FREQ": [], "TF-IDF": [], "LSI": [], "Doc2Vec": []}
    vectors = {"LSI": [], "Doc2Vec": []}
    for entry in ground_truth:
        results, vectors_i = get_results(entry.query, dataframe)
        vectors["LSI"] += vectors_i["LSI"]
        vectors["Doc2Vec"] += vectors_i["Doc2Vec"]
        for query_type in ["FREQ", "TF-IDF", "LSI", "Doc2Vec"]:
            precision = compute_precision(entry, query_type, results)
            scores[query_type].append(Stat(precision, compute_recall(precision)))
    return scores, vectors


def compute_precision(truth, search_type, dataframe):
    counter = 0
    for _, row in dataframe[dataframe['search'] == search_type].iterrows():
        counter += 1
        if row["name"] == truth.name and row["file"] == truth.path:
            return 1 / counter
    return 0


def compute_recall(precision):
    return 1 if precision > 0 else 0


def compute_tsne(dictionary):
    results = {}
    for key, values in dictionary.items():
        tsne = TSNE(n_components=2, verbose=1, perplexity=2, n_iter=3000)
        results[key] = tsne.fit_transform(values)
    return results


def plot_vectors(dictionary, queries):
    for key, values in dictionary.items():
        dataframe = pd.DataFrame()
        dataframe['x'] = values[:, 0]
        dataframe['y'] = values[:, 1]
        plt.figure(figsize=(16, 16))
        plt.title("Results of " + key)

        sns_plot = sns.scatterplot(
            x="x",
            y="y",
            hue=queries + list(itertools.chain.from_iterable([query] * 5 for query in queries)),
            data=dataframe,
            legend="full",
            alpha=1.0
        )
        sns_plot.get_figure().savefig("res/plot_" + key.lower())


def print_scores(scores):
    print("##### PRINT #####")
    for key, values in scores.items():
        print(key)
        precision, recall = compute_mean(values)
        print("\tprecision:\t" + precision)
        print("\trecall:\t\t" + recall)


def compute_mean(stats):
    precision, recall, counter = 0, 0, 0
    for stat in stats:
        precision += stat.precisions
        recall += stat.recalls
        counter += 1
    return str(precision / counter), str(recall / counter)


if len(argv) < 1:
    print("Please give as input ground truth file")
    exit(1)


begin_time = datetime.now()
start(argv[1])
print(datetime.now() - begin_time)
