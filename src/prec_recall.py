from sys import argv, exit
from search_data import get_results
import pandas as pd


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
