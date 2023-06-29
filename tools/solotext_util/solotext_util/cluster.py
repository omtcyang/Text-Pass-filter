import torch
from solotext_util import _C


def clustering(x):
    return _C.clustering(x)


def ker_meaning(label, kernel, column_unique, kernel_index):
    return _C.ker_meaning(label, kernel, column_unique, kernel_index)


def text_counter(mask_connect_labels, all_text_labels):
    return _C.get_masks_counter(mask_connect_labels, all_text_labels)
