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
map_search_radius = 20
lr = 0.001
lr_decay = 0.5
batch_size = 1
epoch = 25
visual = True
TEST_DATA_PATH = '/datastore/data/cxl/argoverse/argo/test_data'
TRAIN_DATA_PATH = '/datastore/data/cxl/argoverse/argo/train_data'
model_save_path = "/datastore/data/cxl/argoverse/argo/model"