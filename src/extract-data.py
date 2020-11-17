from sys import argv, exit
from ast import *
from os import walk
import pandas as pd


class Visitor(NodeVisitor):
    def __init__(self, file_path, node):
        super().__init__()
        self.file_path = file_path
        self.visit(parse(node))

    def visit_ClassDef(self, node: ClassDef):
        self.generic_visit(node)
        if is_valid_entity(node.name):
            self.append_data(node, "class")

    def visit_FunctionDef(self, node: FunctionDef):
        if is_valid_entity(node.name):
            self.generic_visit(node)
            self.append_data(node, "method" if is_method(node) else "function")

    def append_data(self, node, def_type):
        data.append((node.name, self.file_path, node.lineno, def_type, get_docstring(node)))


def is_valid_entity(name):
    return name[0] != '_' and name != "main" and "test" not in name.lower()


def is_method(function):
    return function.args and len(function.args.args) > 0 and 'self' in function.args.args[0].arg


def start(directory_path):
    for path, _, files in walk(directory_path):
        for file_name in files:
            if file_name.endswith('.py'):
                file_path = path + '/' + file_name
                with open(file_path) as file:
                    Visitor(file_path, file.read())

    dataframe = pd.DataFrame(data=data, columns=["name", "file", "line", "type", "comment"])
    dataframe.to_csv('res/data.csv')


if len(argv) < 2:
    print("Please give as input the path of the directory to analyze")
    exit(1)
data = []
start(argv[1])
