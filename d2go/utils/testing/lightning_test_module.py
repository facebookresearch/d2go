#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# pyre-unsafe
from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.utils.data.dataset import Dataset


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class TestModule(LightningModule):
    def __init__(self, epoch_min_loss_override: Optional[int] = None):
        """LightningModule for testing purposes

        Args:
            epoch_min_loss_override (int, optional): Pass in an epoch that will be set to the minimum
                validation loss for testing purposes (zero based). If None this is ignored. Defaults to None.
        """
        super().__init__()
        self.layer = torch.nn.Linear(in_features=32, out_features=2)
        self.another_layer = torch.nn.Linear(in_features=2, out_features=2)
        self.epoch_min_loss_override = epoch_min_loss_override

    def forward(self, x):
        x = self.layer(x)
        return self.another_layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"output": output.detach(), "loss": loss, "checkpoint_on": loss.detach()}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"output": output.detach(), "loss": loss, "checkpoint_on": loss.detach()}

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"output": output.detach(), "loss": loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_loss", avg_loss)

    def validation_epoch_end(self, outputs) -> None:
        avg_val_loss = torch.stack(
            [torch.randn(1, requires_grad=True) for _ in outputs]
        ).mean()
        # For testing purposes allow a nominated epoch to have a low loss
        if self.current_epoch == self.epoch_min_loss_override:
            avg_val_loss -= 1e10
        self.log("val_loss", avg_val_loss)
        self.log("checkpoint_on", avg_val_loss)

    def test_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack(
            [torch.randn(1, requires_grad=True) for _ in outputs]
        ).mean()
        self.log("val_loss", avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(RandomDataset(32, 64))
