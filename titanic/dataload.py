import csv
from itertools import islice


def load_dataset(dataset_file, record_func):
    with open(dataset_file, "rt") as dataset_csv:
        return list(map(record_func, islice(csv.reader(dataset_csv), 1, None)))
