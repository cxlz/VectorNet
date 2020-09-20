import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

time_steps = 49
train_time_steps = 19
pred_time_steps = 30
min_train_time_steps = 5
time_resolution = 0.1
feature_length = 9
lr = 0.001
batch_size = 1
epoch = 25
TEST_DATA_PATH = 'data/argo/test_data/'
TRAIN_DATA_PATH = 'data/argo/train_data/'
model_save_path = "models/model"