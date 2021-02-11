import tensorflow as tf
import numpy as np
import csv
from typing import Optional


class PassRecord:
    def __init__(self, p_id:int, p_class:int, name:str, sex:bool, age:Optional[int], sib_sp:int, parch:int, ticket:str,
                 fare:float, cabin:Optional[str], embarked:Optional[str], survived:Optional[bool]=None):
        self.survived = survived
        self.embarked = embarked
        self.cabin = cabin
        self.fare = fare
        self.ticket = ticket
        self.parch = parch
        self.sib_sp = sib_sp
        self.age = age
        self.sex = sex
        self.name = name
        self.p_class = p_class
        self.p_id = p_id


def train_record(p_id:int, survived:Optional[bool], p_class:int, name:str, sex:bool, age:Optional[int], sib_sp:int, parch:int, ticket:str,
                 fare:float, cabin:Optional[str], embarked:Optional[str]):


def load_dataset(dataset_file, record_func):
    with open(dataset_file, "rb") as dataset_csv:
        return map(lambda line: PassRecord(*line.split(",")),
                   next(csv.reader(dataset_csv)))
        

def load_train_test(train_file, test_file):


