from negative_classification.model import SpecificBERT_Lightning, ConfMatrixLogging
from negative_classification.data import NegClassification_DataModule

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import hydra
from hydra.core.config_store import ConfigStore
from negative_classification.config import Params

cs = ConfigStore.instance()
cs.store(name="params", node=Params)

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

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    dm = NegClassification_DataModule(
        path_to_data=cfg.dataset.path_to_data,
        path_to_bpe_dir=cfg.dataset.path_to_bpe_dir,
        output_dir=working_dir,
        vocab_size=cfg.dataset.vocab_size,
        chunk_lenght=cfg.dataset.chunk_lenght,
        batch_size=cfg.training.batch,
        pad_value=cfg.dataset.pad_value,
        need_to_train_bpe=True,
        test_size_split=cfg.dataset.test_size_split
    )

    model = SpecificBERT_Lightning(
        vocab_size=cfg.dataset.vocab_size,
        embed_dim=cfg.model.embedding_dim,
        pad_value=cfg.dataset.pad_value,
        chunk_lenght=cfg.dataset.chunk_lenght,
        num_classes=cfg.model.num_output_classes,
        depth=cfg.model.layers,
        num_heads=cfg.model.heads,
        mlp_dim=cfg.model.mlp_dim,
        norm_type=cfg.model.norm_type,
        lr=cfg.training.lr,
        qkv_bias=cfg.model.qkv_bias,
        drop_rate=cfg.model.dropout,
        type_of_scheduler = "ReduceOnPlateau", 
        patience_reduce = 20, 
        factor_reduce = 0.1,
        previous_model=None
    )

    wandb.login(key="dec2ee769ce2e455dd463be9b11767cf8190d658")
    wandb_log = WandbLogger(project="NLP_Neg", name="v400-ch50-l8-h4-emb256-mlp512-d0.1-lr3e-4", save_dir=working_dir + "/wandb")

    checkpoint = ModelCheckpoint(
        dirpath=working_dir + "/weights",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    conf_matrix_logger = ConfMatrixLogging(KNOWN_CLASSES)
    early_stop = EarlyStopping(monitor="val_loss", patience=50)

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_log,
        callbacks=[checkpoint, lr_monitor, conf_matrix_logger, early_stop],
        # fast_dev_run=5
    )
    trainer.fit(model=model, datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    L.seed_everything(1702)
    main()