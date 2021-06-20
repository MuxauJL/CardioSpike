import numpy as np
import torch

from dataset import CardioSpikeDataset
from model import CRNN

device = 'cuda:0'


if __name__ == '__main__':
    test_dataset = CardioSpikeDataset('./data/val_no_labels.csv', frame_size=None, test_only=True)

    model = CRNN(lstm_hidden_dim=339, lstm_layers_count=1).cuda()
    checkpoint = torch.load(
        './checkpoints/5_0.8485013623978201_0.9047065659500291_0.8757030371203599_0.9633256196975708.pth')
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        batch_size = 1
        inputs_count = 2
        preds = np.array([])
        for i in range(len(test_dataset)):
            x = test_dataset[i]
            seq_length = len(x)
            x_device = torch.zeros(batch_size, inputs_count, seq_length, dtype=torch.float32, device=device)
            x_device[0, :, :] = x.T
            prediction = model(x_device)
            probs = torch.sigmoid(prediction)
            preds = np.append(preds, (probs > 0.5).detach().cpu())

        results = test_dataset.get_dataframe()
        results['y'] = preds
        results.to_csv('./results.csv')
