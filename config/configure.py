import torch
import os


time_steps = 49
train_time_steps = 19
pred_time_steps = 30
min_train_time_steps = 5
time_resolution = 0.1
feature_length = 9
map_search_radius = 20

TEST_DATA_PATH = 'data/argo/test_data/'
TRAIN_DATA_PATH = 'data/argo/train_data/'
model_save_path = "models/model"
model_save_prefix = "VectorNet_no_slow_obj_"
load_model_path = "models/model/VectorNet_withtimestampe_0922_22:52:46.model"

lr = 0.001
lr_decay = 0.5
batch_size = 32
epoch = 25


visual = True
save_view = True
save_view_path = "data/view/withtimestamp"
if not os.path.exists(save_view_path):
    os.makedirs(save_view_path)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
