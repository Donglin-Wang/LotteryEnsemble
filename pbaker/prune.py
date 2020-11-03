import numpy as np


def flatten_prunable(weights):
    return np.concatenate([layer.reshape(1,-1) for layer in weights if layer.ndim > 1], axis=1)


def unflatten_prunable(flattened, orig_weights):
    curr_offset = 0
    result      = []

    for layer in orig_weights:
        if layer.ndim > 1:
            count = np.prod(layer.shape)
            result.append(flattened[0, curr_offset:curr_offset+count].reshape(layer.shape))
            curr_offset += count
        else:
            result.append(layer)
    return result
