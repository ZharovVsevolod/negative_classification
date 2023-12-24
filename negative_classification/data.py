import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import numpy as np

russian_stopswords = stopwords.words("russian")

def some_prints():
    print(russian_stopswords)
    print()
    print(punctuation)
    print()

def load_dataset(filename: str) -> (np.array, np.array):
    df = pd.read_excel(
        io=filename,
        header=[0],
        index_col=0
    )

    x, y = df["text"].to_numpy(dtype=str), df["class"].to_numpy(dtype=str)
    return x, y


def data_main():
    filepath = "negative_classification/dataset/ds_nlp.xlsx"

    some_prints()
    text, labels = load_dataset(filepath)

    print(labels[:10])
    print()
    print(text[:10])

    print()

    print(np.unique(labels))



if __name__ == "__main__":
    data_main()