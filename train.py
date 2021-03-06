import os
from shutil import copy2
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import CardioSpikeDataset, conv_collate_fn
from metrics import binary_classification_results, \
    binary_classification_metrics_by_counts, calc_sklearn_f1_score
from model import CRNN

device = 'cuda:0'
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def compute_loss(predictions, targets):
    seq_size = targets.shape[0]
    loss = 0
    for i in range(seq_size):
        mask = targets[i] > -0.5
        pos_weight = torch.zeros_like(targets[i])
        pos_weight[mask] = 1
        positives_mask = targets[i] > 0.5
        positives_count = torch.sum(positives_mask)
        if positives_count > 0:
            negatives_mask = targets[i][mask] < 0.5
            coeff = torch.sum(negatives_mask) / positives_count
            pos_weight[positives_mask] *= coeff
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()
        loss += criterion(predictions[i], targets[i])
    return loss


def train_model(model, train_loader, val_loader, optimizer, num_epochs=1, scheduler=None, exp_name=''):
    experiment_path = './exps/' + exp_name
    os.makedirs(experiment_path)
    for file in os.listdir('.'):
        if '.py' in file:
            copy2(file, experiment_path)
    writer = SummaryWriter(log_dir=experiment_path)

    train_loss_history = []
    val_loss_history = []
    val_f1_history = []
    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()

        train_loss_accum = 0
        for i_step, (x, y) in enumerate(train_loader):
            x_device = x.to(device)
            y_device = y.to(device)
            prediction = model(x_device)
            loss_value = compute_loss(prediction, y_device)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            train_loss_accum += loss_value

        train_ave_loss = train_loss_accum / (i_step + 1)
        train_loss_history.append(float(train_ave_loss))

        # validation
        val_freq = 1
        if epoch % val_freq == 0:
            val_loss_accum = 0
            val_total_count = 0
            tp_accum, fp_accum, fn_accum, correct_accum, total_accum = 0, 0, 0, 0, 0
            model.eval()
            with torch.no_grad():
                gts = np.array([])
                preds = np.array([])
                for i_step, (x, y) in enumerate(val_loader):
                    x_device = x.to(device)
                    y_device = y.to(device)
                    prediction = model(x_device)
                    probs = torch.sigmoid(prediction)
                    tp, fp, fn, correct, total = binary_classification_results(
                        probs > 0.5, y_device)
                    tp_accum += tp
                    fp_accum += fp
                    fn_accum += fn
                    correct_accum += correct
                    total_accum += total

                    val_loss_accum += compute_loss(prediction, y_device)
                    val_total_count += 1

                    gts = np.append(gts, y_device.detach().cpu())
                    preds = np.append(preds, (probs > 0.5).detach().cpu())

                precision, recall, f1, accuracy = binary_classification_metrics_by_counts(tp_accum, fp_accum, fn_accum, correct_accum, total_accum)
                print(f'Epoch {epoch + 1}: val precision = {precision}')
                print(f'Epoch {epoch + 1}: val recall = {recall}')
                print(f'Epoch {epoch + 1}: val accuracy = {accuracy}')
                print(f'Epoch {epoch + 1}: val f1 = {f1}')

                f1_micro = calc_sklearn_f1_score(gts, preds, average="micro")
                f1_binary = calc_sklearn_f1_score(gts, preds, average="binary")
                print(f'Epoch {epoch + 1}: sklearn f1 micro = {f1_micro}')
                print(f'Epoch {epoch + 1}: sklearn f1 binary = {f1_binary}')

                val_ave_loss = val_loss_accum / val_total_count
                val_loss_history.append(float(val_ave_loss))
                val_f1_history.append(f1)
                writer.add_scalar('Loss/val', val_ave_loss, epoch + 1)
                writer.add_scalar('Val/F1', f1, epoch + 1)

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), f'{experiment_path}/{epoch + 1}_{precision}_{recall}_{f1}_{accuracy}.pth')

        msg = "Epoch %d: train average loss: %f, val average loss: %f, val f1: %f, val f1 micro: %f, val f1 bin: %f\n" % (
            epoch + 1, train_ave_loss, val_ave_loss, f1, f1_micro, f1_binary)
        print(msg[:-1])
        with open(f'{experiment_path}/log.txt', 'a+') as f:
            f.write(msg)
        writer.add_scalar('Loss/train', train_ave_loss, epoch + 1)

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_ave_loss)
            else:
                scheduler.step()

    return train_loss_history, val_loss_history, val_f1_history


if __name__ == '__main__':
    batch_size = 46
    frame_size = 64
    train_dataset = CardioSpikeDataset('./data/train_split.csv', frame_size=frame_size)
    val_dataset = CardioSpikeDataset('./data/val_split.csv', frame_size=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               collate_fn=conv_collate_fn,
                                               shuffle=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             collate_fn=conv_collate_fn,
                                             num_workers=0)

    model = CRNN(lstm_hidden_dim=339, lstm_layers_count=1).cuda()
    lr = [1e-4, 5e-5]
    weight_decay = [1e-3, 1e-4]
    exp_names = ['crnn_10_339_from_scratch', 'crnn_10_339_finetune']
    num_epochs = [100, 10]
    for train_step in range(2):
        optimizer = optim.AdamW(model.parameters(), lr=lr[train_step],
                                weight_decay=weight_decay[train_step])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995405417351527)

        loss_history, train_history, val_history = train_model(model, train_loader, val_loader, optimizer,
                                                               num_epochs=num_epochs[train_step], scheduler=scheduler,
                                                               exp_name=exp_names[train_step])
        best_score = max(val_history)
        print(f'best F1 score = {best_score}')
