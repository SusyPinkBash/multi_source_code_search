from sys import argv, exit


def start(directory_path):
    pass


if len(argv) < 2:
    print("Please give as input the path of the directory to analyze")
    exit(1)
start(argv[1])
