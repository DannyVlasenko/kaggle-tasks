import tensorflow as tf
import numpy as np
from composable import Composable
from dataload import load_dataset
from typing import Optional


class PassRecord:
    def __init__(self, p_id: int, p_class: int, name: str, sex: bool, age: Optional[int], sib_sp: int, parch: int,
                 ticket: str, fare: float, cabin: Optional[str], embarked: Optional[str],
                 survived: Optional[bool] = None):
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


@Composable
def convert_sex(pos):
    def inner(record_list):
        if record_list[pos].lower() == "male":
            record_list[pos] = 1
        elif record_list[pos].lower() == "female":
            record_list[pos] = 0
        else:
            raise Exception("Wrong sex record in the dataset.")

        return record_list

    return inner


@Composable
def empty_str_none(record_list):
    return map(lambda s: s or None, record_list)


@Composable
def train_record(param_list):
    return PassRecord(param_list[0], *param_list[2:], param_list[1])


@Composable
def test_record(param_list):
    return PassRecord(*param_list)


def train(train_file, test_file):
    test_data = load_dataset(test_file, test_record * (empty_str_none * convert_sex(3)))
    train_data = load_dataset(train_file, train_record * (empty_str_none * convert_sex(4)))


train("data/train.csv", "data/test.csv")
