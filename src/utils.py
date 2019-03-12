import numpy as np


def print_item(item, pos_dict):
    idx = np.argmax(item)
    if idx in pos_dict:
        return pos_dict[idx]
    return '<oov>'


def print_sequence(sequence, pos_dict):
    idx_dict = {v: k for k, v in pos_dict.items()}
    items = np.apply_along_axis(lambda item: print_item(item, idx_dict), 1,
                                sequence)
    items = np.ravel(items)
    print(' '.join(items))
