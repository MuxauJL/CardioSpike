from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import CardioSpikeDataset, conv_collate_fn
from metrics import binary_classification_results, \
    binary_classification_metrics_by_counts, calc_sklearn_f1_score
from model import CRNN

device = 'cuda:0'
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def vis_results(dataset: CardioSpikeDataset, index, probabilities, suptitle=''):
    dataframe_by_index = dataset.get_dataframe_by_index(index)
    gt_positives = dataframe_by_index[dataframe_by_index.y == 1]
    pred_positives = dataframe_by_index[probabilities > 0.5]
    fig, axs = plt.subplots(2)
    fig.suptitle(suptitle)
    axs[0].plot(dataframe_by_index.time, dataframe_by_index.x, color='green')
    axs[0].scatter(gt_positives.time, gt_positives.x, color='red')
    axs[0].set_title('Ground truth')
    axs[1].plot(dataframe_by_index.time, dataframe_by_index.x, color='green')
    axs[1].scatter(pred_positives.time, pred_positives.x, color='red')
    axs[1].set_title('Predictions')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    val_dataset = CardioSpikeDataset('./data/val_split.csv', frame_size=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             collate_fn=conv_collate_fn,
                                             num_workers=0)

    val_total = 0
    val_positives = 0
    for _, y in val_loader:
        mask = y > -0.5
        for label in y[mask]:
            if int(label) == 1:
                val_positives += 1
            val_total += 1
    print(f'Val {val_positives} positives of {val_total} total ({val_positives / val_total * 100} %)')

    model = CRNN(lstm_hidden_dim=339, lstm_layers_count=1).cuda()
    checkpoint = torch.load(
        './checkpoints/5_0.8485013623978201_0.9047065659500291_0.8757030371203599_0.9633256196975708.pth')
    model.load_state_dict(checkpoint)

    tp_accum, fp_accum, fn_accum, correct_accum, total_accum = 0, 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        gts = np.array([])
        preds = np.array([])
        some_results = []
        worst_metric = 1
        for val_index, (x, y) in enumerate(val_loader):
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

            gts = np.append(gts, y_device.detach().cpu())
            preds = np.append(preds, (probs > 0.5).detach().cpu())

            metric_value = calc_sklearn_f1_score(gts[-total:], preds[-total:], average="binary")
            if worst_metric > metric_value:
                worst_metric = metric_value
                some_results.append((val_index, metric_value))

        precision, recall, f1, accuracy = binary_classification_metrics_by_counts(tp_accum, fp_accum, fn_accum,
                                                                                  correct_accum, total_accum)
        print(f'val precision = {precision}')
        print(f'val recall = {recall}')
        print(f'val f1 = {f1}')
        print(f'val accuracy = {accuracy}')

        f1_micro = calc_sklearn_f1_score(gts, preds, average="micro")
        f1_binary = calc_sklearn_f1_score(gts, preds, average="binary")
        print(f'val sklearn f1 micro = {f1_micro}')
        print(f'val sklearn f1 binary = {f1_binary}')

        # visualize some results
        for val_index, metric_value in some_results:
            x, y = conv_collate_fn([val_dataset[val_index]])
            x_device = x.to(device)
            y_device = y.to(device)
            prediction = model(x_device)
            scores = torch.sigmoid(prediction)
            vis_results(val_dataset, val_index, torch.squeeze(scores).detach().cpu().numpy(),
                        suptitle=f'F1 = {metric_value}')
            print(f'F1 = {metric_value}, index = {val_index}, csv_id = {val_dataset.idx_to_id[val_index]}')
