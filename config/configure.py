import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_save_path = "models/model"
TEST_DATA_PATH = 'data/argo/test_data/'
TRAIN_DATA_PATH = 'data/argo/train_data/'