from matplotlib import pyplot as plt
import os


COLORS = ['#008B8B', '#DC143C', '#FFEBCD', '#8A2BE2', '#7FFF00']


def extract_sequences(ampl, labels, time):
    subdata = []
    cur_label = labels[0]
    start_id = 0
    for i, label in enumerate(labels):
        if label != cur_label:
            new_data = {'ampl': ampl[start_id:i + 1],
                        'target': labels[start_id:i + 1],
                        'time': time[start_id: i + 1]}
            cur_label = label
            start_id = i
            subdata.append(new_data)
    subdata.append({'ampl': ampl[start_id:],
                    'target': labels[start_id:],
                    'time': time[start_id:]})

    return subdata


def log_image(result, batch, batch_nb):
    if not os.path.exists('preds'):
        os.makedirs('preds', exist_ok=True)
    len_data = len(batch['time'])
    for i in range(len_data):
        plt.subplot(211)
        subdata = extract_sequences(batch['ampl_unormalized'].detach().cpu().int().numpy()[i, :, 0],
                                         batch['target'].detach().cpu().int().numpy()[i, :, 0],
                                         batch['time_unormalized'].detach().cpu().int().numpy()[i, :, 0])
        for sdata in subdata:
            color = COLORS[sdata['target'][0]]
            plt.plot(sdata['time'], sdata['ampl'], color=color)
        plt.title('GT')
        plt.subplot(212)
        subdata = extract_sequences(batch['ampl_unormalized'].detach().cpu().int().numpy()[i, :, 0],
                                         result.detach().cpu().int().numpy()[i, :, 0],
                                         batch['time_unormalized'].detach().cpu().int().numpy()[i, :, 0])
        for sdata in subdata:
            color = COLORS[sdata['target'][0]]
            plt.plot(sdata['time'], sdata['ampl'], color=color)
        plt.title('Pred')
        plt.savefig(f'preds/img_{batch["id"][i].item()}.jpg')
        plt.close()