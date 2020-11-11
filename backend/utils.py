import numpy as np
import torch


def load_weight(model, torch_path, trainable=True, verbose=False):
    torch_weights = torch.load(torch_path, map_location=torch.device('cpu'))

    # with open(mapping_table_path, 'r') as f:
    #     mapping_table = json.load(f)
    #     mapping_table = {layer['name']: layer['weight'] for layer in mapping_table}

    for layer in model.layers:
        layer.trainable = trainable
        if verbose:
            print(f'Set layer: {layer.name}')
        layer_prefix = layer.name.split('__')[0]
        layer_type = layer.name.split('__')[-1]

        if layer_type == 'conv':
            weight = np.array(torch_weights[f'{layer_prefix}.weight'])
            weight = np.transpose(weight, [2, 3, 1, 0])
            layer.set_weights([weight])
        elif layer_type == 'bn':
            gamma = np.array(torch_weights[f'{layer_prefix}.weight'])
            beta = np.array(torch_weights[f'{layer_prefix}.bias'])
            running_mean = np.array(torch_weights[f'{layer_prefix}.running_mean'])
            running_var = np.array(torch_weights[f'{layer_prefix}.running_var'])
            layer.set_weights([gamma, beta, running_mean, running_var])
        elif layer_type == 'convbias':
            weight = np.array(torch_weights[f'{layer_prefix}.weight'])
            bias = np.array(torch_weights[f'{layer_prefix}.bias'])
            weight = np.transpose(weight, [2, 3, 1, 0])
            layer.set_weights([weight, bias])
        elif layer_type == 'fc':
            weight = np.array(torch_weights[f'{layer_prefix}.weight'])
            bias = np.array(torch_weights[f'{layer_prefix}.bias'])
            layer.set_weights([weight, bias])
        else:
            if verbose:
                print(f'Ignore layer: {layer.name}')
