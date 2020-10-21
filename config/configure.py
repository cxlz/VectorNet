import torch
import os


map_segment_length = 9
time_steps = 49
train_time_steps = 19
pred_time_steps = 30
min_train_time_steps = 5
time_resolution = 0.1
feature_length = 9
map_search_radius = 20

DATA_ROOT = 'data/argo/no_slow_obj'
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'train_data/')
TEST_DATA_PATH =  os.path.join(DATA_ROOT, 'test_data/')
# model_save_path = "models/model"
# model_save_path = "/datastore/data/cxl/VectorNet/models"
model_save_path = "/datastore/data/cxl/VectorNet/models/argo"
model_save_prefix = "VectorNet_test_19_tanh_subgraph_"
train = True
load_model = False 
# load_model_path = "/datastore/data/cxl/VectorNet/models/VectorNet_19_tanh_subgraph_turn_0925_11:11:20.model"
load_model_path = "/datastore/data/cxl/VectorNet/models/our_data/VectorNet_19_tanh_subgraph_1020_18:19:35.model"
# load_model_path = "models/model/VectorNet_no_slow_obj_19_0924_01:57:09.model"
log_file = os.path.join(DATA_ROOT, "train_log.txt")


lr = 0.001
lr_decay = 0.5
batch_size = 32
epoch = 1000


visual = True
save_view = True
save_view_path = "/datastore/data/cxl/VectorNet/view/" + (load_model_path.split("/")[-1]).split(".")[0]# + "/test"
if not os.path.exists(save_view_path):
    os.makedirs(save_view_path)


device = torch.device("cpu")
map_location = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
    map_location = "cuda"
