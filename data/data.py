# from torch.utils import data
#
#
# class MyData(data.Dataset):
#     r"""
#     self.x is all items' feature vectors.  [P, len]
#     self.id is the index of predicted agent. id
#     self.label is the future trajectory vector. [len]
#     """
#     def __init__(self, dataset, isTrain):
#         self.x = dataset
#         self.id = 0
#         self.label = dataset
#         raise NotImplementedError
#
#     def __getitem__(self, index):
#
#
#     def __len__(self):


# import pickle as pkl
#
# with open('data/feature/forecasting_features_test.pkl', "rb") as f:
#     grid_search = pkl.load(f)
#     print(grid_search)
import sys
try:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
except:
    pass
import pickle
import os
from tqdm import tqdm

import pandas as pd
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap
import argparse

X_ID = 3
Y_ID = 4
avm = ArgoverseMap()

def vecLink(a, polyID):
    a = np.array(a)
    ans = []
    type = 0 if a[0, 2] == 'AV' else 1
    for i in range(a.shape[0] - 1):
        l, r = a[i], a[i + 1]
        now = [l[X_ID], l[Y_ID], r[X_ID], r[Y_ID], type,
               l[0],
               r[0],
               np.sqrt(np.square(l[X_ID]-r[X_ID])+np.square(l[Y_ID]-r[Y_ID])) / (r[0]-l[0]),
               polyID]
        ans.append(now)
    return ans


def work(name, file):
    # print("Loading data from file: [%s]"%file)
    ans = pd.read_csv(name)
    ans = np.array(ans)

    city = ans[0][-1]
    track_id = 1

    id = np.argsort(ans[:, 0], kind='mergesort')
    tmp = np.zeros_like(ans)
    for i in range(ans.shape[0]):
        tmp[i] = ans[id[i]]
    ans = tmp

    id = np.argsort(ans[:, 1], kind='mergesort')
    tmp = np.zeros_like(ans)
    for i in range(ans.shape[0]):
        tmp[i] = ans[id[i]]
    ans = tmp

    # print(ans)

    AVX = 0
    AVY = 0
    AVTIME = 0

    tmp = []
    j = 0
    polyID = 0
    for i in range(ans.shape[0]):
        if i + 1 == ans.shape[0] or ans[i, track_id] != ans[i + 1, track_id]:
            now = []
            while j <= i:
                now.append(ans[j])
                if j < i:
                    assert ans[j, 0] <= ans[j + 1, 0]
                j += 1
            vecList = vecLink(now, polyID)
            polyID += 1
            for vec in vecList:
                tmp.append(vec)
            if ans[i, 2] == 'AV':
                AVX, AVY = ans[i-30, 3], ans[i-30, 4]
                distance = np.sqrt(np.square(now[0][3] - now[19][3]) + np.square(now[0][4] - now[19][4]))
                if distance < 1:
                    print(name, distance)
                    return -1
                AVTIME = ans[i-30, 0]

    idList = avm.get_lane_ids_in_xy_bbox(AVX, AVY, city, 20)
    for id in idList:
        lane = avm.city_lane_centerlines_dict[city][id]
        #        print(lane.id)
        #        print(lane.has_traffic_control)
        #        print(lane.turn_direction)
        #        print(lane.is_intersection)
        #        print(lane.centerline)
        ans = []
        for i in range(lane.centerline.shape[0] - 1):
            l, r = lane.centerline[i], lane.centerline[i + 1]

            t = 0
            if lane.turn_direction == 'LEFT':
                t = 1
            elif lane.turn_direction == 'RIGHT':
                t = 2

            now = [l[0], l[1], r[0], r[1], 2,
                   0 if lane.has_traffic_control == False else 1,
                   t,
                   0 if lane.is_intersection == False else 1,
                   polyID]
            tmp.append(now)
            
        polyID += 1

    tmp = np.array(tmp)
    for i in range(tmp.shape[0]):
        tmp[i, 0] -= AVX
        tmp[i, 2] -= AVX
        tmp[i, 1] -= AVY
        tmp[i, 3] -= AVY
        for j in range(4):
            tmp[i , j] *= 100
        if tmp[i, 4] != 2:
            tmp[i, 5] -= AVTIME

    # print(tmp)
    # print(tmp.shape)
    pf = pd.DataFrame(data=tmp)
    pf.to_csv(os.path.join(args.save_dir, 'data_' + file), header=False, index=False)
    return 1


# nameList = ['2645.csv','4791.csv']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="/datastore/data/cxl/argoverse/train/data/", required=False, help="data load dir")
    parser.add_argument("-s", "--save_dir", type=str, default="data/argo/train_data", required=False, help="data save dir")
    parser.add_argument("-n", "--num", type=int, default=5000, required=False, help="num of files to load")
    args = parser.parse_args()
    # DATA_DIR = 'data/argo/forecasting_sample/data/'
    # nameList = ['2645.csv','3700.csv','3828.csv','3861.csv']
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    DATA_DIR = args.data_dir
    nameList = os.listdir(DATA_DIR)
    args.num = min(args.num, len(nameList))
    for i, name in tqdm(enumerate(nameList)):
        if i % 500 == 0 and i != 0:
            print("[%d] data loaded"%i)
        if work(os.path.join(DATA_DIR, name) ,name) > 0:
            args.num -= 1
        if args.num == 0:
            break

# df = pd.read_pickle(FEATURE_DIR + 'forecasting_features_test.pkl')

# feature_idx = [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]
# seq_id = df["SEQUENCE"].values
#
# obs_trajectory = np.stack(
#     df["FEATURES"].values)[:, :20, feature_idx].astype("float")

# print(obs_trajectory.shape)
# print(df.info())
# print(type(df))
# print(df)
#
# print("-------------")
# print(df["SEQUENCE"].values)
# print("-------------")
# print(df["FEATURES"].values)
# print(df["FEATURES"].values.shape)
# print(df["FEATURES"].values[0].shape)
# print(df["FEATURES"].values.shape)
# print(df["FEATURES"].values[0].shape)
# print("-------------")
# print(df["CANDIDATE_CENTERLINES"].values)
# print("-------------")
# print(df["ORACLE_CENTERLINE"].values)
# print("-------------")
# print(df["CANDIDATE_NT_DISTANCES"].values)

# print("-------------")
# print(df['FEATURES'])
# print("-------------")
# print(df["FEATURES"].values)
