from re import finditer
from sys import argv, exit
import pandas as pd
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel, LsiModel, Doc2Vec
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity

##################
def get_results(query, dataframe):
    results_dictionary = compute_results(query, dataframe)
    return pd.DataFrame(data=create_result_dataframe(results_dictionary, dataframe),
                        columns=['name', "file", "line", "type", "comment", "search"])


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


def start_computing(path_csv, path_ground_truth):
    print("##### START #####")
    dataframe = pd.read_csv(path_csv).fillna(value="")
    ground_truth = parse_ground_truth(path_ground_truth)
    scores = compute_precision_recall(ground_truth, dataframe)
    print_scores(scores)


def parse_ground_truth(path_ground_truth):
    print("##### GROUND TRUTH #####")
    classes = []
    for entry in open(path_ground_truth, "r").read().split("\n\n"):
        data = entry.split("\n")
        classes.append(Truth(data[0], data[1], data[2]))
    return classes


def compute_precision_recall(ground_truth, dataframe):
    print("##### COMPUTE #####")
    scores = {"FREQ": [], "TF-IDF": [], "LSI": [], "Doc2Vec": []}
    for entry in ground_truth:
        results = get_results(entry.query, dataframe)
        for query_type in ["FREQ", "TF-IDF", "LSI", "Doc2Vec"]:
            precision = compute_precision(entry, query_type, results)
            scores[query_type].append(Stat(precision, compute_recall(precision)))
    return scores


def compute_precision(truth, search_type, dataframe):
    precision, counter = 0, 0
    for _, row in dataframe[dataframe['search'] == search_type].iterrows():
        if row["name"] == truth.name and row["file"] == truth.path:
            return 1 / (counter + 1)
        counter += 1
    return precision


def compute_recall(precision):
    return 1 if precision > 0 else 0


def print_scores(scores):
    print("##### PRINT #####")
    for key, values in scores.items():
        print(key)
        print("\tprecision:\t" + compute_mean([stat.precisions for stat in values]))
        print("\trecall:\t" + compute_mean([stat.recalls for stat in values]))


def compute_mean(arrg):
    return str(sum(arrg) / len(arrg))


if len(argv) < 2:
    print("Please give as input the csv of search data and the ground truth files")
    exit(1)

print("#########################")
start_computing(argv[1], argv[2])
