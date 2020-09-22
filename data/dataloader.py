import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data

from argoverse.map_representation.map_api import ArgoverseMap

import config.configure as config
from data.data import mapVec, City

# TEST_DATA_PATH = 'data/data/'
# TRAIN_DATA_PATH = 'data/data/'
# TRAIN_FILE = ['2645.csv','3700.csv']
# TEST_FILE = ['3828.csv','3861.csv','4791.csv']


TEST_DATA_PATH = config.TEST_DATA_PATH
TRAIN_DATA_PATH = config.TRAIN_DATA_PATH

# for root, dirs, files in os.walk(TEST_DATA_PATH):
#     TEST_FILE = files

# for root, dirs, files in os.walk(TRAIN_DATA_PATH):
#     TRAIN_FILE = files
TRAIN_FILE = []
TEST_FILE = []

for file in os.listdir(TRAIN_DATA_PATH):
    if file.split("_")[0] == "obj":
        TRAIN_FILE.append(file)
for file in os.listdir(TEST_DATA_PATH):
    if file.split("_")[0] == "obj":
        TEST_FILE.append(file)
TRAIN_FILE.sort()
TEST_FILE.sort()

time_steps = config.time_steps
train_time_steps = config.train_time_steps

r"""
data structure:
  [start_X, start_Y, end_X, end_Y, type, att1, att2, att3, polyline_id]
  
  type: 0 for predicted agent, 1 for other agents, 2 for lane.
  att:
    For agents:
      'att1' and 'att2' are time stamp of begin and end points.
      'att3' is speed. 
    For lanes:
      'att1' represents whether it has traffic control.
      'att2' represents it's direction. 0 is NONE, 1 and 2 are LEFT and RIGHT respectively.
      'att3' represents whether it is a intersection.
    
"""
avm = ArgoverseMap()

def load_data(DATA_PATH, nameList):
    r"""
    Loading data from files.
    :param nameList: the files for generating.
    :return: X and Y represents input and label.
    """

    polyline_ID = 8
    type_ID = 4
    maxSize = np.zeros(500)
    offset = []
    XX = []
    YY = []
    for name in tqdm(nameList):
        ans = pd.read_csv(os.path.join(DATA_PATH, name), header=None)
        ans = np.array(ans)
        head_line = ans[0]
        city = City(head_line[0]).name
        ans = ans[1:]
        # map_name = name.replace("obj", "map")
        # map_data = pd.read_csv(DATA_PATH + map_name, header=None)
        # map_data = np.array(map_data)
        obj_num = round(ans.shape[0] / time_steps) 
        X = np.zeros((0, config.feature_length))
        Y = np.zeros((0, 2))
        data_len = [0]
        label_len = [0]
        for i in range(obj_num):
            cur_pos = i * time_steps
            data_pos = cur_pos + train_time_steps
            label_pos = cur_pos + time_steps
            data = ans[cur_pos:data_pos, :]
            label = ans[data_pos:label_pos, :]
            idx = data[:, 2] != 0
            data = data[idx, :]
            data_len.append(data_len[-1] + data.shape[0])
            idx = label[:, 2] != 0
            label = label[idx, :]
            label_len.append(label_len[-1] + label.shape[0])
            X = np.concatenate((X, data))
            Y = np.concatenate((Y, label[:, 2:4]))
        # X = np.concatenate((X, map_data))
        xx = []
        yy = []
        for i in range(obj_num):
            if data_len[i + 1] - data_len[i] > config.min_train_time_steps \
                and label_len[i + 1] - label_len[i] == config.pred_time_steps:
                x = X.copy()
                y = Y[label_len[i]:label_len[i + 1]].copy()
                AVX = x[data_len[i + 1] - 1, 2]
                AVY = x[data_len[i + 1] - 1, 3]
                map_data = mapVec(AVX, AVY, city, config.map_search_radius, obj_num, avm)
                x = np.concatenate((x, map_data), axis=0)
                x[:, 0:3:2] -= AVX
                x[:, 1:4:2] -= AVY
                y[:, 0] -= AVX
                y[:, 1] -= AVY
                maxX = np.max(np.abs(x[data_len[i]:data_len[i + 1], 0:3:2]))
                maxY = np.max(np.abs(x[data_len[i]:data_len[i + 1], 1:4:2]))
                # if maxX == np or maxY == 0 or np.isnan(maxX) or np.isnan(maxY):
                #     continue
                x[:, 0:3:2] /= maxX
                x[:, 1:4:2] /= maxY
                y[:, 0] /= maxX
                y[:, 1] /= maxY
                tmp = np.zeros((1, 9))
                tmp[0, 0] = i
                tmp[0, 1] = AVX
                tmp[0, 2] = AVY
                tmp[0, 3] = maxX
                tmp[0, 4] = maxY
                x = np.concatenate((tmp, x), axis=0)
                y = np.reshape(y, -1)
                tmp = np.zeros(config.pred_time_steps * 2 - y.shape[0])
                y = np.concatenate((y, tmp), axis=0)
                xx.append(x)
                yy.append(y)
        max_feat_num = -1
        for i in range(len(xx)):
            if len(xx[i]) > max_feat_num:
                max_feat_num = len(xx[i])
        zerolist = np.zeros((1, config.feature_length))
        for i in range(len(xx)):
            zerolist[0, -1] = xx[i][0, -1]
            for j in range(len(xx[i]), max_feat_num):
                xx[i] = np.concatenate((xx[i], zerolist), axis=0)
        XX.append(np.array(xx))
        YY.append(np.array(yy))
    return XX, YY

    #     x, tx, y = [], [], []
    #     j = 0

    #     maxX, maxY = 0, 0
    #     for i in range(ans.shape[0]):
    #         if ans[i, type_ID] == 0:
    #             maxX = np.max([maxX, np.abs(ans[i, 0]), np.abs(ans[i, 2])])
    #             maxY = np.max([maxY, np.abs(ans[i, 1]), np.abs(ans[i, 3])])

    #     for i in range(ans.shape[0]):
    #         if ans[i, type_ID] != 2:
    #             ans[i, 5] = ans[i, 6] = ans[i, 7] = 0

    #     dx, dy = 1, 1
    #     for i in range(ans.shape[0]):
    #         if i + 1 == ans.shape[0] or \
    #                 ans[i, polyline_ID] != ans[i + 1, polyline_ID]:
    #             id = int(ans[i, polyline_ID])
    #             # if ans[i, type_ID] == 2:
    #             #     j = i + 1
    #             #     continue
    #             if ans[i, type_ID] == 0:  # predicted agent
    #                 t = np.zeros_like(ans[0]).astype('float')
    #                 t[0] = ans[i, polyline_ID]
    #                 x.append(t)
    #                 # print(i)

    #                 # if i-j+1 != 49:
    #                 #     print(DATA_PATH + 'data_' + name)

    #                 assert i - j + 1 == 49
    #                 maxSize[id] = np.max([maxSize[id], 19])
    #                 if ans[j, 0] > 0:
    #                     dx = -1
    #                 if ans[j, 1] > 0:
    #                     dy = -1

    #                 for l in range(0, 19):
    #                     tx.append(ans[j])
    #                     j += 1
    #                 for l in range(19, 49):
    #                     y.append(ans[j, 2])
    #                     y.append(ans[j, 3])
    #                     j += 1
    #             else:
    #                 maxSize[id] = np.max([maxSize[id], i - j + 1])
    #                 while j <= i:
    #                     tx.append(ans[j])
    #                     j += 1
    #     print(dx, dy, name)

    #     for xx in tx:
    #         xx[0] *= dx
    #         xx[2] *= dx
    #         xx[1] *= dy
    #         xx[3] *= dy
    #         xx[0] /= maxX
    #         xx[2] /= maxX
    #         xx[1] /= maxY
    #         xx[3] /= maxY
    #         x.append(xx)
    #     for i in range(0, len(y), 2):
    #         y[i] *= dx
    #         y[i + 1] *= dy
    #         y[i] /= maxX
    #         y[i + 1] /= maxY

    #     offset.append([0, 0, 0, 0, 0, maxX, maxY, 0, 0])
    #     x = np.array(x).astype('float')
    #     y = np.array(y).astype('float')

    #     # print(x.shape)

    #     X.append(x)
    #     Y.append(y)
    # pf = pd.DataFrame(data=x)
    # pf.to_csv('train_data_' + name, header=False, index=False)

    # ans = 0
    # for i in range(0, maxSize.shape[0]):
    #     ans += maxSize[i]

    # # print(ans)
    # XX = []
    # YY = Y
    # for it in range(len(X)):
    #     x = []
    #     x.append(X[it][0])
    #     j = 1
    #     for i in range(0, maxSize.shape[0]):
    #         if maxSize[i] == 0:
    #             break
    #         tmp = maxSize[i]
    #         lst = np.zeros(9)
    #         lst[polyline_ID] = i
    #         while j < X[it].shape[0] and X[it][j, polyline_ID] == i:
    #             x.append(X[it][j])
    #             lst = X[it][j]
    #             j += 1
    #             tmp -= 1
    #         while tmp > 0:
    #             x.append(lst)
    #             tmp -= 1
    #     XX.append(x)
    # pf = pd.DataFrame(data=XX[0])
    # pf.to_csv('train_data_XX_' + name, header=False, index=False)
    # for i in range(len(offset)):
    #     XX[i].append(offset[i])
    # XX = np.array(XX).astype('float')
    # YY = np.array(YY).astype('float')

    # # print(XX)

    # # print(XX.shape)
    # # print(YY.shape)
    # # for i in range(XX.shape[1]):
    # #     print(XX[0,i,polyline_ID],XX[1,i,polyline_ID])
    # # exit(0)

    # XX = torch.from_numpy(XX)
    # YY = torch.from_numpy(YY)

    # XX = XX.float()
    # YY = YY.float()

    # train = torch.utils.data.TensorDataset(XX, YY)
    # return train


class ArgoDataset(data.Dataset):
    def __init__(self, DATA_PATH, nameList):
        self.DATA_PATH = DATA_PATH
        self.nameList = nameList
        self.X, self.Y = load_data(self.DATA_PATH, self.nameList)

    def __getitem__(self,index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


def load_train():
    r"""
    Loading train set.
    :return: train set.
    """
    return load_data(TRAIN_DATA_PATH, TRAIN_FILE)


def load_test():
    r"""
    Loading test set.
    :return: test set.
    """
    return load_data(TEST_DATA_PATH, TEST_FILE)

# if __name__ == '__main__':
#     load_train()
# np_arr = np.array([[1], [2], [3], [4]])
# tor_arr = torch.from_numpy(np_arr)
# print(type(np_arr))
