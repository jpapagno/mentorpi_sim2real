"""Shared network layout for training and lightweight policy playback."""

HIDDEN_DIMS = [128, 128]
CNN_CONFIG = {
    "output_channels": [8, 16, 16],
    "kernel_size": [8, 4, 3],
    "stride": [4, 2, 1],
    "activation": "elu",
}
