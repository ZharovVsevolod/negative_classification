from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from torch import nn
from torch.nn import functional as F
import einops
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, ConfusionMatrix, MulticlassF1Score
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import onnxruntime as ort
import youtokentome as yttm
from data import text_lemmatize
from pymystem3 import Mystem
import re

KNOWN_CLASSES = [
    'Вежливость сотрудников магазина',
    'Время ожидания у кассы',
    'Доступность персонала в магазине',
    'Компетентность продавцов/ консультантов',
    'Консультация КЦ',
    'Обслуживание на кассе',
    'Обслуживание продавцами/ консультантами',
    'Электронная очередь'
]

def get_result(text: pd.Series) -> pd.Series:
    # Модель в формате onnx
    save_onnx = "negative_classification/models/self_model.onnx"
    ort_session = ort.InferenceSession(save_onnx)
    answers = []

    # Предобработка
    np_text = text.to_numpy()
    # print(np_text)

    ## Лемматизатор
    mystem = Mystem()
    lemma_text = text_lemmatize(np_text, mystem)

    ## Токенизатор
    tokenizer = yttm.BPE("negative_classification/models/bpe_300.yttm")

    for phrase in lemma_text:
        phrase_in_tokens = text_preprocessing(phrase, tokenizer)
        phrase_in_tokens = einops.rearrange(phrase_in_tokens, "chunk -> 1 chunk")
        asnwer_class = get_answer_from_model(
            input_sample=phrase_in_tokens,
            ort_session=ort_session
        )
        answers.append(asnwer_class)

    return pd.Series(answers, name="class_predicted")

def text_preprocessing(text: np.array, tokenizer):
    # Очистка
    text = " ".join(text)
    text = re.findall("[а-яА-Я ]+", text)
    text = " ".join(text)

    # Токенизация
    text_tokens = tokenizer.encode(text, bos=True, eos=True)
    text_tokens = ensure_lenght(text_tokens)
    return np.array(text_tokens)

def ensure_lenght(txt, chunk_lenght=60, pad_value=0):
        if len(txt) < chunk_lenght:
            txt = list(txt) + [pad_value]*(chunk_lenght - len(txt))
        else:
            txt = txt[:chunk_lenght]
        return txt

def get_answer_from_model(input_sample, ort_session: ort.InferenceSession) -> str:
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_sample}

    ort_outs = ort_session.run(None, ort_inputs)

    tensor_outputs = torch.from_numpy(np.array(ort_outs)).squeeze()[-1, :]
    answer_class = torch.argmax(tensor_outputs)
    return KNOWN_CLASSES[answer_class]


class Class_Positions_Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size, pad_value, chunk) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk = chunk
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_value
        )
        self.positional_embedding = nn.Parameter(torch.rand(1, self.chunk, self.embed_dim))
        self.class_tokens = nn.Parameter(torch.rand(1, 1, self.embed_dim))

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_embedding.data
        batch, chunk, emd_dim = x.shape
        class_token = einops.repeat(self.class_tokens.data, "() chunk dim -> batch chunk dim", batch=batch)
        x = torch.cat((x, class_token), dim=1)
        return x

class MLP(nn.Module):
    def __init__(self, in_features:int, hidden_features=None, out_features=None, drop=0.0, act_layer = nn.GELU()):
        super().__init__()
        if out_features is None:
            out_features = in_features
        if hidden_features is None:
            hidden_features = in_features

        # Linear Layers
        self.lin1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.lin2 = nn.Linear(
            in_features=hidden_features,
            out_features=out_features
        )

        # Activation(s)
        self.act = act_layer
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.act(self.dropout(self.lin1(x)))
        x = self.act(self.lin2(x))

        return x

class Attention(nn.Module):
    def __init__(self, dim:int, num_heads:int, qkv_bias=False, attn_drop=0.0, out_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.soft = nn.Softmax(dim=-1) # Softmax по строкам матрицы внимания
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x):
        # Attention
        qkv_after_linear = self.qkv(x)
        qkv_after_reshape = einops.rearrange(qkv_after_linear, "b c (v h w) -> v b h c w", v=3, h=self.num_heads)
        q = qkv_after_reshape[0]
        k = qkv_after_reshape[1]
        k = einops.rearrange(k, "b h c w -> b h w c") # Транспонирование
        v = qkv_after_reshape[2]

        atten = self.soft(torch.matmul(q, k) * self.scale)
        atten = self.attn_drop(atten)
        out = torch.matmul(atten, v)
        out = einops.rearrange(out, "b h c w -> b c (h w)", h=self.num_heads)

        # Out projection
        x = self.out(out)
        x = self.out_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim:int, norm_type:str, num_heads:int, mlp_dim:int, qkv_bias=False, drop_rate=0.0):
        super().__init__()
        self.norm_type = norm_type

        # Normalization
        self.norm1 = nn.LayerNorm(
            normalized_shape=dim
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=dim
        )

        # Attention
        self.attention = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            out_drop=drop_rate
        )
        
        # MLP
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_dim
        )


    def forward(self, x):
        if self.norm_type == "prenorm":
            x_inner = self.norm1(x)
            # Attetnion
            x_inner = self.attention(x_inner)
            x = x_inner + x

            x_inner = self.norm2(x)
            # MLP
            x_inner = self.mlp(x_inner)
            x = x_inner + x
        
        if self.norm_type == "postnorm":
            x_inner = self.attention(x)
            x = x_inner + x
            x = self.norm1(x)
            x_inner = self.mlp(x)
            x = x_inner + x
            x =self.norm2(x)

        return x

class Transformer(nn.Module):
    def __init__(self, depth, dim, norm_type, num_heads, mlp_dim, qkv_bias=False, drop_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, norm_type, num_heads, mlp_dim, qkv_bias, drop_rate) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SpecificBERT(nn.Module):
    def __init__(
            self, 
            vocab_size, embed_dim, pad_value, chunk_lenght,
            num_classes, depth, num_heads, mlp_dim,
            norm_type,
            qkv_bias=False, drop_rate=0.0
        ):
        super().__init__()
        # Позиционное кодирование и Эмбеддинги + Токен класса
        self.class_pos_emb = Class_Positions_Embeddings(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            pad_value=pad_value,
            chunk=chunk_lenght
        )
        
        # Transformer Encoder
        self.transformer = Transformer(
            depth=depth,
            dim=embed_dim,
            norm_type=norm_type,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate
        )
        # Classifier
        self.head = MLP(
            in_features=embed_dim,
            out_features=num_classes,
            drop=drop_rate
        )

    def forward(self, x):
        x = self.class_pos_emb(x)
        x = self.transformer(x)
        x = self.head(x)
        return x

class SpecificBERT_Lightning(L.LightningModule):
    def __init__(
            self,
            vocab_size:int, embed_dim:int, pad_value:int, chunk_lenght:int,
            num_classes:int, depth:int, num_heads:int, mlp_dim:int,
            norm_type:str,
            lr:float,
            qkv_bias=False, drop_rate=0.0,
            type_of_scheduler:str = "ReduceOnPlateau", patience_reduce:int = 5, factor_reduce:float=0.1, 
            # lr_coef_cycle:int = 2, total_num_of_epochs:int = 20,
            previous_model = None
        ) -> None:
        super().__init__()
        if previous_model is None:
            self.bert_model = SpecificBERT(
                vocab_size=vocab_size, embed_dim=embed_dim, pad_value=pad_value, chunk_lenght=chunk_lenght,
                num_classes=num_classes, depth=depth, num_heads=num_heads, mlp_dim=mlp_dim,
                norm_type=norm_type, qkv_bias=qkv_bias, drop_rate=drop_rate
            )
        else:
            self.bert_model = previous_model
        
        self.metric_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.metric_f1 = MulticlassF1Score(num_classes=num_classes)
        self.matrix = ConfusionMatrix(task = "multiclass", num_classes = num_classes)
        self.flag_conf_matrix = True
        self.num_classes = num_classes

        self.lr = lr
        self.type_of_scheduler = type_of_scheduler
        self.patience_reduce = patience_reduce
        self.factor_reduce = factor_reduce
        # self.lr_coef_cycle = lr_coef_cycle
        # self.total_num_of_epochs = total_num_of_epochs
        

        self.save_hyperparameters()

    def forward(self, x):
        return self.bert_model(x)
    
    def loss(self, y, y_hat):
        return F.cross_entropy(y, y_hat)

    def lr_scheduler(self, optimizer):
        if self.type_of_scheduler == "ReduceOnPlateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience_reduce, factor=self.factor_reduce)
            scheduler_out = {"scheduler": sched, "monitor": "val_loss"}
        # if self.type_of_scheduler == "OneCycleLR":
        #     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr * self.lr_coef_cycle, total_steps=self.total_num_of_epochs)
        #     scheduler_out = {"scheduler": sched}
        
        return scheduler_out
    
    def log_everything(self, pred_loss, out, y, name:str):
        self.log(f"{name}_loss", pred_loss)
        self.log(f"{name}_acc", self.metric_accuracy(out, y))
        self.log(f"{name}_f1", self.metric_f1(out, y))

    def training_step(self, batch) -> STEP_OUTPUT:
        x, y = batch
        x = torch.stack(x)
        x = einops.rearrange(x, "len batch -> batch len")

        out = self(x)[:,-1,:]
        pred_loss = self.loss(out, y)

        self.log_everything(pred_loss, out, y, name="train")
        
        return pred_loss
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        x, y = batch
        x = torch.stack(x)
        x = einops.rearrange(x, "len batch -> batch len")

        out = self(x)[:,-1,:]
        pred_loss = self.loss(out, y)

        self.log_everything(pred_loss, out, y, name="val")

        if self.flag_conf_matrix:
            self.conf_matrix = self.matrix(torch.softmax(out, dim=-1), y)
            self.flag_conf_matrix = False
        else:
            self.conf_matrix += self.matrix(torch.softmax(out, dim=-1), y)
    
    def test_step(self, batch) -> STEP_OUTPUT:
        pass
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        scheduler_dict = self.lr_scheduler(optimizer)
        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
        )

class ConfMatrixLogging(L.Callback):
    def __init__(self, cls) -> None:
        super().__init__()
        self.cls = cls
    
    def make_img_matrix(self, matr):
        matr = matr.cpu()
        fig=plt.figure(figsize=(16, 8), dpi=80)
        plt.imshow(matr,  interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()

        tick_marks = np.arange(len(self.cls))
        plt.xticks(tick_marks, self.cls, rotation=90)
        plt.yticks(tick_marks, self.cls)

        fmt = 'd'
        thresh = matr.max() / 2.
        for i, j in itertools.product(range(matr.shape[0]), range(matr.shape[1])):
            plt.text(j, i, format(matr[i, j], fmt), horizontalalignment="center", color="white" if matr[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.show()
        return [fig]

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        trainer.logger.log_image(key="Validation Confusion Matrix", images=self.make_img_matrix(pl_module.conf_matrix))
        plt.close()
        pl_module.flag_conf_matrix = True