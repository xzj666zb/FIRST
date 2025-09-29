import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NUM_EPOCHS_POLARIZED = 20
NUM_EPOCHS_FINETUNE = 20

TRAIN_DATA_PATH = "data/train"
VAL_DATA_PATH = "data/test"
TEST_DATA_PATH ="data/test"
LOG_SAVE_PATH = "logs"

MODEL_SAVE_PATH = "models/saved_models"


PRINT_EVERY = 1