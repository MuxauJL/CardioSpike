import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim

from data import CardioSpikeDataset, collate_fn, conv_collate_fn
from model import BiLSTMDetector

device = 'cuda:0'
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def vis_time_series_by_id(data, id):
    time_series = data[data.id == id]
    time_series_positive = time_series[time_series.y == 1]
    plt.figure(figsize=(15, 7))
    plt.plot(time_series.time, time_series.x, color='green')
    plt.scatter(time_series_positive.time, time_series_positive.x, color='red')
    plt.grid(True)
    plt.show()


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction (seq_length, batch_size, 1) - model predictions
    ground_truth (seq_length, batch_size, 1) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    true_positives_count = 0
    false_positives_count = 0
    false_negatives_count = 0
    correct_count = 0
    mask = ground_truth > -1
    total_count = torch.sum(mask)
    gt = torch.flatten(ground_truth[mask])
    pred = torch.flatten(prediction[mask])
    for j in range(total_count):
        if gt[j]:
            if pred[j]:
                true_positives_count += 1
                correct_count += 1
            else:
                false_negatives_count += 1
        else:
            if pred[j]:
                false_positives_count += 1
            else:
                correct_count += 1
    if true_positives_count + false_positives_count > 0:
        precision = true_positives_count / (true_positives_count + false_positives_count)
    else:
        precision = 1
    if true_positives_count + false_negatives_count > 0:
        recall = true_positives_count / (true_positives_count + false_negatives_count)
    else:
        recall = 1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    accuracy = correct_count / total_count

    return precision, recall, f1, accuracy


def compute_loss(predictions, targets):
    seq_size = len(targets)
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


def train_model(model, train_loader, val_loader, optimizer, num_epochs=1, scheduler=None):
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train()

        loss_accum = 0
        for i_step, (x, y) in enumerate(train_loader):
            x_device = x.to(device)
            y_device = y.to(device)
            prediction = model(x_device)
            loss_value = compute_loss(prediction, torch.unsqueeze(y_device, dim=2))
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_accum += loss_value

        ave_loss = loss_accum / i_step

        loss_history.append(float(ave_loss))

        # validation
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x_device = x.to(device)
                y_device = y.to(device)
                prediction = model(x_device)
                probs = torch.sigmoid(prediction)
                precision, recall, f1, accuracy = binary_classification_metrics(
                    probs > 0.5, torch.unsqueeze(y_device, dim=2))
                print(f'Epoch {epoch + 1}: val precision = {precision}')
                print(f'Epoch {epoch + 1}: val recall = {recall}')
                print(f'Epoch {epoch + 1}: val f1 = {f1}')
                print(f'Epoch {epoch + 1}: val accuracy = {accuracy}')

        val_history.append(f1)
        print("Epoch %d: average loss: %f, val f1: %f" % (
            epoch + 1, ave_loss, f1))

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss_history[-1])
            else:
                scheduler.step()

    return loss_history, train_history, val_history


if __name__ == '__main__':
    # train_data = pd.read_csv('C:/Users/Mikhail/Desktop/Хахатон/CardioSpike/data/train.csv')

    # train_data = train_data.to_numpy(dtype=np.int64)
    # print(train_data.shape)
    # for idx, time, x, y in train_data:
    #     exit(0)

    # id = 12
    # vis_time_series_by_id(train_data, id)
    # exit(0)

    # # cross-validation
    # num_folds = 5
    # train_data = train_data.to_numpy(dtype=np.int64)
    # folds_X = np.array_split(train_data[:, 1:-1], num_folds)
    # folds_y = np.array_split(train_data[:, -1], num_folds)
    #
    # print(f'total positives: {np.sum(train_data[:, -1] == 1) / len(train_data[:, -1])}')
    # for i in range(num_folds):
    #     train_X_fold = np.concatenate([fold for j, fold in enumerate(folds_X) if i != j])
    #     train_y_fold = np.concatenate([fold for j, fold in enumerate(folds_y) if i != j])
    #     print(f'{i} train fold positives: {np.sum(train_y_fold == 1) / len(train_y_fold)}')
    #     print(f'{i} val fold positives: {np.sum(folds_y[i] == 1) / len(folds_y[i])}')
    # exit(0)
    #
    # for i in range(num_folds):
    #     train_X_fold = np.concatenate([fold for j, fold in enumerate(folds_X) if i != j])
    #     train_y_fold = np.concatenate([fold for j, fold in enumerate(folds_y) if i != j])

    batch_size = 46
    num_folds = 5
    train_dataset = CardioSpikeDataset('C:/Users/Mikhail/Desktop/Хахатон/CardioSpike/data/train.csv')
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    # folds_indices = np.array_split(indices, num_folds)
    # for i in range(num_folds):
    #     train_indices = np.concatenate([idx for j, idx in enumerate(folds_indices) if i != j])
    #     train_sampler = SubsetRandomSampler(train_indices)
    #     val_sampler = SubsetRandomSampler(folds_indices[i])
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
    #                                                sampler=train_sampler, collate_fn=collate_fn)
    #     val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(folds_indices[i]),
    #                                              sampler=val_sampler, collate_fn=collate_fn)

    validation_fraction = .2
    val_split = int(np.floor(validation_fraction * len(train_dataset)))
    val_indices, train_indices = indices[:val_split], indices[val_split:]
    # np.save('val.npy', val_indices)
    # np.save('train.npy', train_indices)
    # import json
    # with open("idx_to_id.json", "w") as write_file:
    #     json.dump(train_dataset.idx_to_id, write_file)
    # exit(0)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler, collate_fn=conv_collate_fn)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(val_indices),
                                             sampler=val_sampler, collate_fn=conv_collate_fn)
    # train_total = 0
    # train_positives = 0
    # for _, y in train_loader:
    #     mask = y > -0.5
    #     for label in y[mask]:
    #         if int(label) == 1:
    #             train_positives += 1
    #         train_total += 1
    # print(f'Train {train_positives} positives of {train_total} total ({train_positives / train_total * 100} %)')
    #
    # val_total = 0
    # val_positives = 0
    # for _, y in val_loader:
    #     mask = y > -0.5
    #     for label in y[mask]:
    #         if int(label) == 1:
    #             val_positives += 1
    #         val_total += 1
    # print(f'Val {val_positives} positives of {val_total} total ({val_positives / val_total * 100} %)')
    # exit(0)

    lstm_hidden_dims = [128]
    max_score = 0
    for lstm_hidden_dim in lstm_hidden_dims:
        model = BiLSTMDetector(lstm_hidden_dim=lstm_hidden_dim).cuda()
        optimizer = optim.AdamW(model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        loss_history, train_history, val_history = train_model(model, train_loader, val_loader, optimizer,
                                                               num_epochs=20, scheduler=scheduler)

        best_score = max(val_history)
        print(f'best score with {lstm_hidden_dims} channels: {best_score}')
        if best_score > max_score:
            best = lstm_hidden_dims
            max_score = best_score

    print(f'Best out_channels = {best}')
    print(f'Best score = {max_score}')