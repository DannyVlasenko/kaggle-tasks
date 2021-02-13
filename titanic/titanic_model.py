import csv
from itertools import islice
from composable import Composable
from typing import Optional
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
            record_list[pos] = "1"
        elif record_list[pos].lower() == "female":
            record_list[pos] = "0"
        else:
            raise Exception("Wrong sex record in the dataset.")

        return record_list

    return inner


@Composable
def empty_str_none(record_list):
    return map(lambda s: s or None, record_list)


@Composable
def train_record(param_iter):
    param_list = list(param_iter)
    return PassRecord(param_list[0], *param_list[2:], param_list[1])


@Composable
def test_record(param_iter):
    param_list = list(param_iter)
    return PassRecord(*param_list)


def split_train_record(rec: PassRecord):
    return float(rec.survived), np.array([rec.p_class, rec.age, rec.sex, rec.sib_sp, rec.parch], dtype=float)


def split_test_record(rec: PassRecord):
    return int(rec.p_id), np.array([rec.p_class, rec.age, rec.sex, rec.sib_sp, rec.parch], dtype=float)


def median_age(records_iter):
    median = np.median(list(map(lambda rec: float(rec.age), filter(lambda rec: rec.age is not None, records_iter))))

    def substitute_age(record):
        if record.age is None:
            record.age = median
            return record
        else:
            return record

    return map(substitute_age, records_iter)


def prepare_data(csv_lines, record_func, split_func):
    records_iter = median_age(list(map(record_func, csv_lines)))
    label_iter, rec_iter = zip(*map(split_func, records_iter))
    return np.array(label_iter), np.array(rec_iter)


def load_data(data_file, record_func, split_func):
    with open(data_file, "rt") as data_csv:
        return prepare_data(islice(csv.reader(data_csv), 1, None), record_func, split_func)


train_record_func = train_record * (empty_str_none * convert_sex(4))
test_record_func = test_record * (empty_str_none * convert_sex(3))


class AccCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["accuracy"] > 0.90:
            self.model.stop_training = True


def train(train_file):
    train_labels, train_records = load_data(train_file, train_record_func, split_train_record)
    print(train_labels.shape, train_records.shape)
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(128, activation="relu", input_dim=5),
                                        tf.keras.layers.Dense(32, activation="relu"),
                                        tf.keras.layers.Dense(1, activation="sigmoid")])

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_records, train_labels, epochs=40, callbacks=[AccCallback()])
    model.summary()
    return model, history


def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy, loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()


def predict(test_file, model, result_file):
    def survived_transform(value):
        if value > 0.5:
            return 1
        else:
            return 0

    ids, test_records = load_data(test_file, test_record_func, split_test_record)
    survived = map(survived_transform, model.predict(test_records))
    with open(result_file, "wt") as result_csv:
        result_csv.write("PassengerId,Survived\n")
        result_csv.writelines(map(lambda pid, val: str(pid) + "," + str(val) + "\n", ids, survived))
