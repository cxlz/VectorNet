import os

from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import config.configure as config
from torch.utils import data

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
TRAIN_FILE = sorted(os.listdir(TRAIN_DATA_PATH))
TEST_FILE = sorted(os.listdir(TEST_DATA_PATH))
# TRAIN_FILE = np.random.choice(TRAIN_FILE, min(len(TRAIN_FILE), 128))
# TEST_FILE = np.random.choice(TEST_FILE, min(len(TEST_FILE), 50))

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


def load_data(DATA_PATH, nameList):
    r"""
    Loading data from files.
    :param nameList: the files for generating.
    :return: X and Y represents input and label.
    """

    X = []
    MX = []
    Y = []
    polyline_ID = 8
    type_ID = 4
    max_obj_size = np.zeros(1000)
    max_map_size = np.zeros(1000)
    max_m_id = 0
    offset = []
    m_id = []
    for name in nameList:
        ans = pd.read_csv(DATA_PATH + name, header=None)
        ans = np.array(ans)
        x, mx, tx, mtx, y = [], [], [], [], []
        j = 0

        maxX, maxY = 0, 0
        for i in range(ans.shape[0]):
            if ans[i, type_ID] == 0:
                maxX = np.max([maxX, np.abs(ans[i, 0]), np.abs(ans[i, 2])])
                maxY = np.max([maxY, np.abs(ans[i, 1]), np.abs(ans[i, 3])])

        # for i in range(ans.shape[0]):
        #     if ans[i, type_ID] != 2:
        #         ans[i, 5] = ans[i, 6] = ans[i, 7] = 0

        dx, dy = 1, 1
        cur_m_id = 0
        for i in range(ans.shape[0]):
            if i + 1 == ans.shape[0] or \
                    ans[i, polyline_ID] != ans[i + 1, polyline_ID]:
                if ans[i, type_ID] == 2 and cur_m_id == 0:
                    cur_m_id = int(ans[i, polyline_ID])
                    max_m_id = max(max_m_id, cur_m_id)
                    m_id.append(cur_m_id)
                id = int(ans[i, polyline_ID]) - cur_m_id
                # if ans[i, type_ID] == 2:
                #     j = i + 1
                #     continue
                if ans[i, type_ID] == 0:  # predicted agent
                    t = np.zeros_like(ans[0]).astype('float')
                    t[0] = ans[i, polyline_ID]
                    x.append(t)
                    # print(i)

                    # if i-j+1 != 49:
                    #     print(DATA_PATH + 'data_' + name)

                    if i - j + 1 != 49:
                        break
                    max_obj_size[id] = np.max([max_obj_size[id], 19])
                    if ans[j, 0] > 0:
                        dx = -1
                    if ans[j, 1] > 0:
                        dy = -1

                    for l in range(0, 19):
                        tx.append(ans[j])
                        j += 1
                    for l in range(19, 49):
                        y.append(ans[j, 2])
                        y.append(ans[j, 3])
                        j += 1
                else:
                    count = 0
                    if ans[i, type_ID] == 2:
                        max_map_size[id] = np.max([max_map_size[id], i - j + 1])
                        while j <= i:
                            mtx.append(ans[j])  
                            j += 1
                    else:
                        max_obj_size[id] = np.max([max_obj_size[id], min(19, i - j + 1)])
                        while j <= i:
                            if count < 19:
                                tx.append(ans[j])
                                count += 1   
                            j += 1
        # print(dx, dy, name)

        for xx in tx:
            xx[0] *= dx
            xx[2] *= dx
            xx[1] *= dy
            xx[3] *= dy
            xx[0] /= maxX
            xx[2] /= maxX
            xx[1] /= maxY
            xx[3] /= maxY
            x.append(xx)
        for xx in mtx:
            xx[0] *= dx
            xx[2] *= dx
            xx[1] *= dy
            xx[3] *= dy
            xx[0] /= maxX
            xx[2] /= maxX
            xx[1] /= maxY
            xx[3] /= maxY
            mx.append(xx)
        for i in range(0, len(y), 2):
            y[i] *= dx
            y[i + 1] *= dy
            y[i] /= maxX
            y[i + 1] /= maxY
        x[0][3] = maxX
        x[0][4] = maxY
        # x[0][5] = m_id[-1]
        offset.append([0, 0, 0, 0, 0, maxX, maxY, 0, 0])
        x = np.array(x).astype('float')
        mx = np.array(mx).astype('float')
        y = np.array(y).astype('float')

        # print(x.shape)

        X.append(x)
        MX.append(mx)
        Y.append(y)
    # pf = pd.DataFrame(data=x)
    # pf.to_csv('train_data_' + name, header=False, index=False)

    ans = 0
    for i in range(0, max_obj_size.shape[0]):
        ans += max_obj_size[i]

    # print(ans)
    XX = []
    MXX = []
    YY = Y
    for it in range(len(X)):
        x = []
        X[it][0, 5] = max_m_id
        x.append(X[it][0])
        j = 1
        for i in range(0, max_obj_size.shape[0]):
            if max_obj_size[i] == 0:
                break
            tmp = max_obj_size[i]
            lst = np.zeros(9)
            lst[polyline_ID] = i
            while j < X[it].shape[0] and X[it][j, polyline_ID] == i:
                x.append(X[it][j])
                lst = X[it][j]
                tmp -= 1
                j += 1
            while tmp > 0:
                x.append(lst)
                tmp -= 1
        XX.append(x)
    for it in range(len(MX)):
        mx = []
        j = 0
        for i in range(0, max_map_size.shape[0]):
            cur_m_id = i + m_id[it]
            if max_map_size[i] == 0:
                break
            tmp = max_map_size[i]
            lst = np.zeros(9)
            lst[polyline_ID] = cur_m_id
            while j < MX[it].shape[0] and MX[it][j, polyline_ID] == cur_m_id:
                MX[it][j, polyline_ID] = max_m_id + i
                mx.append(MX[it][j])
                lst = MX[it][j]
                tmp -= 1
                j += 1
            while tmp > 0:
                mx.append(lst)
                tmp -= 1
            
        MXX.append(mx)
    # pf = pd.DataFrame(data=XX[0])
    # pf.to_csv('train_data_XX_' + name, header=False, index=False)
    for i in range(len(offset)):
        XX[i].append(offset[i])
    XX = np.array(XX).astype('float')
    MXX = np.array(MXX).astype('float')
    XX = np.concatenate((XX, MXX), axis=1)
    YY = np.array(YY).astype('float')

    # print(XX)

    # print(XX.shape)
    # print(YY.shape)
    # for i in range(XX.shape[1]):
    #     print(XX[0,i,polyline_ID],XX[1,i,polyline_ID])
    # exit(0)

    XX = torch.from_numpy(XX).float()
    YY = torch.from_numpy(YY).float()

    # XX = XX.float()
    # YY = YY.float()
    return XX, YY
    # train = torch.utils.data.TensorDataset(XX, YY)
    # return train


def load_train():
    r"""
    Loading train set.
    :return: train set.
    """
    return load_data(TRAIN_DATA_PATH, TRAIN_FILE[:500])


def load_test():
    r"""
    Loading test set.
    :return: test set.
    """
    return load_data(TEST_DATA_PATH, TEST_FILE)


def my_collate_fn(batch_data):
    DATA_PATH = batch_data[0][0]
    nameList = []
    for path, name in batch_data:
        nameList.append(name)
    X, Y = load_data(DATA_PATH, nameList)
    return X, Y


class ArgoDataset(data.Dataset):
    def __init__(self, DATA_PATH, nameList):
        self.DATA_PATH = DATA_PATH
        self.nameList = nameList
        # self.X, self.Y = load_data(self.DATA_PATH, self.nameList)

    def __getitem__(self,index):
        return self.DATA_PATH, self.nameList[index]
        # return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.nameList)
# if __name__ == '__main__':
#     load_train()
# np_arr = np.array([[1], [2], [3], [4]])
# tor_arr = torch.from_numpy(np_arr)
# print(type(np_arr))
