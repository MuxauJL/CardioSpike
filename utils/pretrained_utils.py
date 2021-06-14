import torch


def load_model(model, pretrained_checkpoint_path):
    pretrained_model = torch.load(pretrained_checkpoint_path, map_location='cpu')
    model.load_state_dict(load_diff_pretrained(model.state_dict(), pretrained_model['state_dict']))

def load_diff_pretrained(new_state_dict, pretrained_state_dict):

    for name, param in new_state_dict.items():
        if name in pretrained_state_dict:
            input_param = pretrained_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                print('Shape mismatch at:', name, 'skipping')
        else:
            print(f'{name} weight of the model not in pretrained weights')
    return new_state_dict