from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import pytorch_lightning as pl

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nn.vae import VAE
from adabelief_pytorch import AdaBelief


class LitVAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim,
        kl_beta,
        learning_rate,
        weight_decay=1e-4,
        log_interval=50,
        log_val_interval=5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.configure_plotter()

        self.vae = VAE(self.hparams.latent_dim)
        self.metrics = {}

    def configure_optimizers(self):
        opt = AdaBelief(
            self.vae.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            print_change_log=False
        )
        # opt = torch.optim.AdamW(
        #     self.vae.parameters(),
        #     lr=self.hparams.learning_rate,
        #     weight_decay=self.hparams.weight_decay,
        # )
        return opt

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx)

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        **kwargs,
    ):
        """
        run at each step for training or validation
        """
        # extract data and run model
        inputs = batch["x"]
        targets = inputs

        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        loss_logs = {
            f"{name}_{['val', 'train'][self.training]}": value
            for name, value in loss.items()
        }
        self.log_dict(loss_logs, prog_bar=True, logger=True)

        self.log_metrics(batch, outputs)
        if self.log_interval > 0:
            self.log_prediction(batch, outputs, batch_idx)

        return loss["loss"]

    def criterion(self, outputs, targets):
        reconstruct, mu, log_var = outputs

        r_loss = F.mse_loss(reconstruct, targets)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )
        loss = r_loss + kl_loss * self.hparams.kl_beta
        return {"loss": loss, "loss_r": r_loss, "loss_kl": -kl_loss}

    def log_metrics(self, *args, **kwargs):
        pass

    def plot_prediction(
        self,
        batch: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
        idx: int = 0,
        add_loss_to_title: bool = False,
    ) -> plt.Figure:
        """
        Plot prediction of prediction vs actuals
        """
        target = batch["x"][idx, 0].detach().cpu().numpy()
        prediction = output[0][idx, 0].detach().cpu().numpy()

        fig, ax = plt.subplots()
        MSE = np.mean((target - prediction) ** 2)
        ax.set_title(f"Prediction Plot\n(MSE={MSE:.3f})")

        xticks = np.arange(0, len(target))
        ax.plot(xticks, target, label="target")
        ax.plot(xticks, prediction, label="prediction")
        ax.legend()
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def log_prediction(
        self,
        batch: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Log metrics every training/validation step.
        """
        # log single prediction figure
        if (
            batch_idx % self.log_interval == 0 or self.log_interval < 1.0
        ) and self.log_interval > 0:
            if self.log_interval < 1.0:  # log multiple steps
                log_indices = torch.arange(
                    0,
                    self.hparams.batch_size,
                    max(1, round(self.log_interval * self.hparams.batch_size)),
                )
            else:
                log_indices = [0]

            for idx in log_indices:
                fig = self.plot_prediction(
                    batch, output, idx=idx, add_loss_to_title=True
                )
                tag = f"{['Val', 'Train'][self.training]} prediction"
                tag += f" of item {idx} in batch {batch_idx}"

                if isinstance(fig, (list, tuple)):
                    for idx, f in enumerate(fig):
                        self.logger.experiment.add_figure(
                            f"Target {idx} {tag}",
                            f,
                            global_step=self.global_step,
                        )
                        plt.close(f)
                else:
                    self.logger.experiment.add_figure(
                        tag,
                        fig,
                        global_step=self.global_step,
                    )
                    plt.close(fig)

    @property
    def log_interval(self) -> float:
        """
        Log interval depending if training or validating
        """
        if self.training:
            return self.hparams.log_interval
        else:
            return self.hparams.log_val_interval

    def configure_plotter(self):
        DPI = 200
        mpl.rc("savefig", dpi=DPI)
        mpl.rcParams["figure.dpi"] = DPI
        mpl.rcParams["figure.figsize"] = 6.4, 4.8  # Default.
        mpl.rcParams["font.sans-serif"] = "Roboto"
        mpl.rcParams["font.family"] = "sans-serif"

        # Set title text color to dark gray (https://material.io/color) not black.
        TITLE_COLOR = "#212121"
        mpl.rcParams["text.color"] = TITLE_COLOR

        rc = {
            # "axes.spines.left": False,
            # "axes.spines.right": False,
            # "axes.spines.bottom": False,
            # "axes.spines.top": False,
            # "xtick.bottom": False,
            # "xtick.labelbottom": False,
            # "ytick.labelleft": False,
            # "ytick.left": False,
        }
        mpl.rcParams.update(rc)
