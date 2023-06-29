import torch
from solotext_util2 import solotext_util_ext


def text_counter(mask_connect_labels, all_text_labels):
    return solotext_util_ext.get_masks_counter(mask_connect_labels, all_text_labels)
