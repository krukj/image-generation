DATA_DIR = "data/cats/CAT_0"
DATA_PROCESSED_DIR = "data_output/cats"
DATA_PROCESSED_TORCH_DIR = "data_output"
MARGIN_RATIO = 0.3
TARGET_SIZE = 64

NGPU = 1
BATCH_SIZE = 128
WORKERS = 2
RESULTS_PATH = "results"

DCGAN_DEFAULT = {
    "nz": 128,
    "d_to_g_ratio": 1,
    "batch_size": 128,
    "lr": 0.0002,
    "beta": 0.5,
    "num_epochs": 5,
}

DCGAN_GRID = {"nz": [32, 128, 512], "d_to_g_ratio": [2, 3, 5]}
