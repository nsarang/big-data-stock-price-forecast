import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import ccxt
from datetime import datetime, timedelta, timezone

from core.data.preprocess import process_dtypes, add_time_idx
from core.data.cdd import get_crypto_dataset
from core.data.dataset import StockDataset
from core.nn.lit import LitVAE


def load_datasets(symbols, timeframe, val_ratio, sequence_length):
    # LOAD DAT
    exchange = ccxt.bitstamp({"enableRateLimit": True, "rateLimit": 85})

    trains, vals = [], []
    for sym in symbols:
        df = get_crypto_dataset(
            exchange,
            symbol=sym,
            timeframe=timeframe,
            start_date=datetime(2015, 1, 1),
            max_per_page=1000,
        )
        df = process_dtypes(df)
        df = add_time_idx(df)
        df = df.iloc[len(df) // 10 :]
        print(sym, len(df))

        val_size = int(len(df) * val_ratio)
        train, val = StockDataset(df[:-val_size], sequence_length), StockDataset(
            df[-val_size:], sequence_length
        )
        trains.append(train)
        vals.append(val)

    train_dataset = ConcatDataset(trains)
    val_dataset = ConcatDataset(vals)
    return train_dataset, val_dataset


if __name__ == "__main__":
    # CONFIG
    sequence_length = 256
    val_ratio = 0.15
    timeframe = "1h"
    symbols = [
        "BTC/USD",
        "ETH/USD",
        "ETH/BTC",
        "LTC/BTC",
        "LTC/USD",
        "XRP/USD",
        "XRP/BTC",
    ]

    batch_size = 128
    num_workers = 4

    experiment_name = "08-low-beta-ckpt"
    latent_dim = 32
    kl_beta = 0.0005
    learning_rate = 1e-4
    num_sanity_val_steps = 10
    max_epochs = 400

    # DATA
    train_dataset, val_dataset = load_datasets(
        symbols, timeframe, val_ratio, sequence_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # TRAIN
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="loss_val",
        dirpath=f"checkpoints/{experiment_name}",
        filename="masknet-{epoch:02d}-{loss_val:.2f}",
        save_top_k=5,
        mode="min",
    )
    logger = pl.loggers.TensorBoardLogger("logs", name=experiment_name)

    model = LitVAE(latent_dim, kl_beta, learning_rate)

    num_gpus = torch.cuda.device_count()
    accelerator = "ddp" if num_gpus else None

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=logger,
        gpus=num_gpus,
        accelerator=accelerator,
        num_sanity_val_steps=num_sanity_val_steps,
        max_epochs=max_epochs,
        flush_logs_every_n_steps=10,
        progress_bar_refresh_rate=5,
        weights_summary="full",
    )

    trainer.fit(model, train_loader, val_loader)
