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
from utils.vis_utils import log_image


class CovidCardioSpikeExperiment(pl.LightningModule):

    def __init__(self,
                 hparams: Namespace) -> None:
        super(CovidCardioSpikeExperiment, self).__init__()

        self.save_hyperparameters(hparams)

        self.num_classes = self.hparams.num_classes
        self.create_model()
        print(self.net)
        self.loss = registry.CRITERION.get_from_params(**self.hparams.train.loss_args)

    def get_scheduler(self, optimizer):
        args = {**self.hparams.train.scheduler_params}
        args['optimizer'] = optimizer

        return registries.SCHEDULERS.get_from_params(**args)

    def configure_optimizers(self):
        print('configure parameters')
        params = filter(lambda p: p.requires_grad, self.net.parameters())
        optimizer = registries.OPTIMIZERS.get_from_params(**{'params': params, **self.hparams.train.optimizer_params})

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': self.get_scheduler(optimizer), 'interval': 'epoch'}}

    # Extract mask where falses will be more than trues by coef times
    def negative_sampling_mask(self, labels, mask, coef=3):
        with torch.no_grad():
            N,C,H = labels.shape
            positives = torch.sum(labels, dim=[-1,-2])
            assert len(positives.shape) == 1, positives.shape
            positives_mask = labels[:,:, 0] > 0.5
            new_mask = torch.zeros_like(mask)
            for i in range(N):
                elements = mask[i].sum()
                sampling_matrix = torch.rand(elements).type_as(labels) < coef * positives[i] / elements
                new_mask[i, :elements] = sampling_matrix
                new_mask[i] = new_mask[i] | positives_mask[i]
        return new_mask

    def attend_on_bad_preds_mask(self, prediction, labels, mask, thres=0.4, min_elems=20):
        #return torch.not_equal(prediction > 0.5, labels > 0.5) & mask.unsqueeze(2)
        new_mask = torch.not_equal(prediction > 0.5 + thres, labels > 0.5) & torch.not_equal(prediction < 0.5 - thres, labels < 0.5) & mask.unsqueeze(2)
        if new_mask.sum() < min_elems:
            print('return default mask')
            new_mask = mask
        return new_mask

    def create_model(self):
        opt = self.hparams
        print('opt', opt.keys())
        self.net = networks.define_model(opt.network)

    def forward(self, input):
        keys = self.hparams.input_keys
        new_input = {v: input[k] for k, v in keys.items()}
        return self.net(**new_input)

    def forward_loss(self, prediction, target, mask):
        if self.hparams.train.loss_args.criterion == 'LocalBetterLoss':
            return self.loss(prediction, target, mask)
        else:
            return self.loss(prediction[mask], target[mask])


    def training_step(self, batch, batch_idx):
        output = self(batch)
        if self.num_classes == 1:
            if self.hparams.train.use_negative_sampling:
                mask = self.negative_sampling_mask(batch['target'], batch['mask_bool'])
            else:
                mask = batch['mask_bool']
            loss = self.forward_loss(output, batch['target'], mask)
        else:
            assert self.hparams.train.use_negative_sampling == False
            target = torch.argmax(batch['target'], dim=-1)[batch['mask_bool']]
            #print(output[batch['mask_bool']].shape, target[(target < 0) | (target >= self.num_classes)], target.shape, torch.argmax(batch['target'], dim=-1).shape)
            loss = self.forward_loss(output, target, batch['mask_bool'])

        self.log('loss', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):

        def extract_int_numpy(tensor):
            return tensor.detach().cpu().int().numpy()

        if self.num_classes > 1:
            scores = self(batch)
            pred = torch.argmax(scores, dim=2, keepdim=True)
            labels = torch.argmax(batch['target'], dim=2, keepdim=True)
            scores = torch.softmax(scores, 2)
        else:
            result = self(batch)
            if self.hparams.train.use_negative_sampling:
                mask = self.negative_sampling_mask(batch['target'], batch['mask_bool'])
            else:
                mask = batch['mask_bool']
            loss = self.forward_loss(result, batch['target'], mask)
            self.log('val_loss', loss, prog_bar=True)
            result = torch.sigmoid(result)
            scores = result
            pred = (result > self.hparams.threshold).int()
            labels = batch['target']


        if self.hparams.train.use_plt:
            log_image(result=extract_int_numpy(pred), target=extract_int_numpy(labels), time=extract_int_numpy(batch['time_unormalized']), ampl=extract_int_numpy(batch['ampl_unormalized']), id=batch["id"])


        mask = batch['mask_bool']

        return {'pred': pred.detach().cpu().numpy(), 'target': labels.cpu().numpy(),
                'score': scores.detach().cpu().numpy(), 'mask': batch['mask_bool'].cpu().numpy()}


    def validation_epoch_end(self, outputs):
        concated_pred = []
        concated_target = []
        for batch in outputs:
            pred, score, target, mask = batch['pred'], batch['score'], batch['target'], batch['mask']
            for i, el in enumerate(pred):
                concated_pred.append(el[mask[i]].reshape(-1))
                concated_target.append(target[i][mask[i]].reshape(-1))

        concated_pred = np.concatenate(concated_pred)
        concated_target = np.concatenate(concated_target)

        f1_score = metrics.f1_score(concated_target, concated_pred, pos_label=1, average='binary' if self.num_classes == 1 else 'micro')
        f1_score_micro = metrics.f1_score(concated_target, concated_pred, pos_label=1, average='micro')


        self.log('f1_score', f1_score, prog_bar=True)
        self.log('f1_score_micro', f1_score_micro, prog_bar=True)

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
                          batch_size=val_params.batch_size,
                          shuffle=self.hparams.train.val_steps_limit < 1.0 or type(self.hparams.train.val_steps_limit) == int,
                          drop_last=False,
                          num_workers=1, collate_fn=self.val_dataset.collate_fn)

    def train_dataloader(self):
        train_params = self.hparams.datasets.train
        return DataLoader(self.train_dataset,
                          batch_size=train_params.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=train_params.n_workers, collate_fn=self.val_dataset.collate_fn)
