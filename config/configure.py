import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

time_steps = 49
train_time_steps = 19
time_resolution = 0.1
TEST_DATA_PATH = 'data/argo/test_data/'
TRAIN_DATA_PATH = 'data/argo/train_data/'
model_save_path = "models/model"