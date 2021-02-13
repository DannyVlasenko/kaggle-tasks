from titanic_model import train
from titanic_model import plot_history
from titanic_model import predict


if __name__ == '__main__':
    model, history = train("data/train.csv")
    predict("data/test.csv", model, "data/result.csv")
    plot_history(history)

