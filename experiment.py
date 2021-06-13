from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
from torch import optim, autograd

from torch.utils.data import DataLoader
import torch.nn as nn

import registry
from dataset import create_dataset

from models import networks

from registry import registries
from transforms.transform import get_test_transform, get_train_transform
import torch
import sklearn.metrics as metrics
from utils.confusion_matrix import plot_confusion_matrix, render_figure_to_tensor


class CovidCardioSpikeExperiment(pl.LightningModule):

    def __init__(self,
                 hparams: Namespace) -> None:
        super(CovidCardioSpikeExperiment, self).__init__()

        self.save_hyperparameters(hparams)

        self.num_classes = self.hparams.num_classes
        self.create_model()
        self.loss = registry.CRITERION.get_from_params(**self.hparams.train.loss_args)

    def get_scheduler(self, optimizer):
        args = {**self.hparams.train.scheduler_params}
        args['optimizer'] = optimizer

        return registries.SCHEDULERS.get_from_params(**args)

    def configure_optimizers(self):
        params = self.net.parameters()
        optimizer = registries.OPTIMIZERS.get_from_params(**{'params': params, **self.hparams.train.optimizer_params})

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': self.get_scheduler(optimizer), 'interval': 'epoch'}}

    def create_model(self):
        opt = self.hparams
        print('opt', opt.keys())
        self.net = networks.define_model(opt.network)

    def forward(self, input):
        keys = self.hparams.input_keys
        new_input = {v: input[k] for k, v in keys.items()}
        return self.net(**new_input)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(output, batch['target'])

        log = {'loss': loss}
        return {'loss': loss, 'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_nb):
        if self.num_classes > 1:
            scores = self(batch)
            pred = torch.argmax(scores, dim=1)

            scores = torch.softmax(scores, 1)
        else:
            result = torch.sigmoid(self(batch))
            scores = result
            pred = (result.squeeze(1) > self.hparams.threshold).int()

        labels = batch['target']
        mask = batch['mask_bool']

        return {'pred': pred.detach().cpu().numpy(), 'target': labels.cpu().numpy(),
                'score': scores.detach().cpu().numpy(), 'mask': batch['mask_bool'].cpu().numpy()}


    def validation_epoch_end(self, outputs):
        f1_scores = []
        for batch in outputs:
            pred, score, target, mask = batch['pred'], batch['score'], batch['target'], batch['mask']
            for i, el in enumerate(pred):
                f1_score = metrics.f1_score(target[i][mask[i]].reshape(-1), el[mask[i]].reshape(-1), pos_label=1, average='binary')
                f1_scores.append(f1_score)

        self.log('f1_score', np.mean(f1_scores))

    def prepare_data(self):
        mode = self.hparams.datasets.mode

        val_params = self.hparams.datasets.val
        val_transform = get_test_transform(val_params.augs_args)
        self.val_dataset = create_dataset(mode, val_params.dataset_args, val_transform)

        train_params = self.hparams.datasets.train
        train_transform = get_train_transform(train_params.augs_args)
        self.train_dataset = create_dataset(mode, train_params.dataset_args, train_transform)

    def val_dataloader(self):
        val_params = self.hparams.datasets.val
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          drop_last=False,
                          num_workers=1, collate_fn=self.val_dataset.collate_fn)

    def train_dataloader(self):
        train_params = self.hparams.datasets.train
        return DataLoader(self.train_dataset,
                          batch_size=train_params.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=train_params.n_workers, collate_fn=self.val_dataset.collate_fn)
