from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import youtokentome as yttm
from sklearn.model_selection import train_test_split
import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
import re
import os

def load_text_from_file(filename):
    with open(filename) as f:
        lines = [line[:-1] for line in f.readlines()]
    return lines

def load_dataset(filename: str) -> (np.array, np.array):
    df = pd.read_excel(
        io=filename,
        header=[0],
        index_col=0
    )
    x, y = df["text"].to_numpy(dtype=str), df["class"].to_numpy(dtype=str)
    return x, y

def text_lemmatize(text, stopwrd = stopwords.words("russian")):
    mystem = Mystem()
    lemma_texts = [mystem.lemmatize(phrase) for phrase in text]
    answer = []
    for phrase in lemma_texts:
        temp_phrase = [token for token in phrase if token not in stopwrd and token != " " and token.strip() not in punctuation]
        answer.append(temp_phrase)
    return answer

def for_txt_file(texts, filename):
    file = open(filename, "x")
    for phrase in texts:
        phr = " ".join(phrase)
        phr = re.findall("[а-яА-Я ]+", phr)
        phr = " ".join(phr) + "\n"
        file.write(phr)
    file.close()

class LanduageModelDataset(Dataset):
    def __init__(self, text_tokens, labels, chunk_lenght=60, pad_value=0) -> None:
        self.text_tokens = text_tokens
        self.labels = labels
        self.chunk_lenght = chunk_lenght
        self.pad_value = pad_value
    
    def __len__(self):
        return len(self.labels)
    
    def ensure_lenght(self, txt):
        if len(txt) < self.chunk_lenght:
            txt = list(txt) + [self.pad_value]*(self.chunk_lenght - len(txt))
        else:
            txt = txt[:self.chunk_lenght]
        return txt

    def __getitem__(self, index):
        text = self.text_tokens[index]
        text = self.ensure_lenght(text)
        label = self.labels[index]
        return text, label
    
class NegClassification_DataModule(L.LightningDataModule):
    def __init__(
            self, 
            path_to_data, 
            path_to_bpe_dir, 
            output_dir, 
            vocab_size, 
            chunk_lenght, 
            batch_size, 
            pad_value, 
            need_to_train_bpe=True, 
            test_size_split=0.1
        ) -> None:
        super().__init__()
        self.path_to_bpe_dir = path_to_bpe_dir
        self.path_to_data = path_to_data
        self.output_dir = output_dir + "/dataset"
        self.vocab_size = vocab_size
        self.need_to_train_bpe = need_to_train_bpe
        self.test_size_split = test_size_split
        self.chunk_lenght = chunk_lenght
        self.pad_value = pad_value
        self.batch_size = batch_size
    
    def preprocessing(self):
        # Создание папки
        os.mkdir(self.output_dir)
        # Загрузка данных
        text, labels = load_dataset(self.path_to_data)

        # Создание словаря-переводчика меток класса и перевод классов в числа
        unique_labels = np.unique(labels)
        self.label_to_num = dict()
        for idx in range(len(unique_labels)):
            self.label_to_num[unique_labels[idx]] = idx
        
        self.filename_label_txt = self.output_dir + "/labels.txt"
        file = open(self.filename_label_txt, "x")
        for lbl in labels:
            file.write(lbl + "\n")
        file.close()
        
        # Лемматизация
        lemma_texts = text_lemmatize(text)
        self.filename_lemma = self.output_dir + "/neg_text.txt"
        for_txt_file(lemma_texts, self.filename_lemma)

        # Словарь BPE
        BPE_MODEL_FILENAME = self.path_to_bpe_dir + f"/bpe_{self.vocab_size}.yttm"
        if self.need_to_train_bpe:
            yttm.BPE.train(data=self.filename_lemma, vocab_size=self.vocab_size, model=BPE_MODEL_FILENAME)
        self.tokenizer = yttm.BPE(BPE_MODEL_FILENAME)
    
    def prepare_data(self) -> None:
        self.preprocessing()
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            # Загрузка предобработанных данных
            lemma_text = load_text_from_file(self.filename_lemma)
            labels = load_text_from_file(self.filename_label_txt)
            num_labels = [self.label_to_num[label] for label in labels]

            # Деление на тренировочную и тестовую выборку и токенизация
            train_text, val_text, train_labels, val_labels = train_test_split(lemma_text, num_labels, test_size=self.test_size_split)
            train_tokens = self.tokenizer.encode(train_text, bos=True, eos=True)
            val_tokens = self.tokenizer.encode(val_text, bos=True, eos=True)

            # Созание датасетов
            self.train_dataset = LanduageModelDataset(
                text_tokens=train_tokens,
                labels=train_labels,
                chunk_lenght=self.chunk_lenght,
                pad_value=self.pad_value
            )
            self.val_dataset = LanduageModelDataset(
                text_tokens=val_tokens,
                labels=val_labels,
                chunk_lenght=self.chunk_lenght,
                pad_value=self.pad_value
            )

        if stage == "test" or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass