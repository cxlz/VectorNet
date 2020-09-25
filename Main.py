import argparse
import numpy as np
import sys
try:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
except:
    pass
import os
import cv2
import time

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchnet import meter

from models.VectorNet import VectorNet
from models.VectorNet import VectorNetWithPredicting
# from config.configure import device
from data.dataloader import load_train, load_test, ArgoDataset, my_collate_fn
import config.configure as config

device = config.device
TEST_DATA_PATH = config.TEST_DATA_PATH
TRAIN_DATA_PATH = config.TRAIN_DATA_PATH

# for root, dirs, files in os.walk(TEST_DATA_PATH):
#     TEST_FILE = files

# for root, dirs, files in os.walk(TRAIN_DATA_PATH):
#     TRAIN_FILE = files
TRAIN_FILE = sorted(os.listdir(TRAIN_DATA_PATH))
TEST_FILE = sorted(os.listdir(TEST_DATA_PATH))

def askADE(a, b):
    r"""
    Calculate Average Displacement Error(ADE).
    :param a:
    :param b:
    :return: ADE
    """
    # print('loss')
    # print(a)
    # print(b)
    ans = torch.zeros(1).to(device)
    minDis = torch.zeros(1).to(device)
    minDis[0] = float('inf')
    for it in range(a.shape[0]):
        tmp = 0
        for i in range(0, a.shape[1], 2):
            x1, y1 = a[it, i], a[it, i + 1]
            x2, y2 = b[it, i], b[it, i + 1]
            tmp += torch.sqrt(torch.square(x1 - x2).to(device)
                              + torch.square(y1 - y2).to(device))
        endDis = torch.sqrt(torch.square(a[it, -1] - b[it, -1]).to(device)
                            + torch.square(a[it, -2] - b[it, -2]).to(device))
        tmp /= a.shape[1] / 2
        if minDis > endDis:
            minDis = endDis
            ans = tmp
        print(endDis.item(), tmp)
        # ans += tmp
    return [minDis, ans]


# def askLoss(a, b):
#     # print('loss')
#     # print(a.shape)
#     # print(b)
#     ans = torch.zeros(1).to(device)
#     for it in range(a.shape[0]):
#         for i in range(0, a.shape[1], 2):
#             x1, y1 = a[it, i], a[it, i + 1]
#             x2, y2 = b[it, i], b[it, i + 1]
#             ans += torch.sqrt(torch.square(x1 - x2).to(device) + torch.square(y1 - y2).to(device)).to(device)
#     ans /= a.shape[0] * a.shape[1] / 2
#
#     return ans


def random_train(epoch, learningRate, batchSize):
    r"""
    A simple test, demonstrate my VectorNet is working.
    If you want to see convergence in training, please set the epoch lager and delete the code
    'learningRate *= 0.7'.
    :param epoch:
    :param learningRate:
    :param batchSize:
    :return:
    """
    timeStampNumber = 2
    len = 5
    train_X = torch.tensor(
        [[[1, 0, 0, 0, 0],
          [1, 2, 3, 0, 0],
          [0, 0, 1, 1, 1],
          [1, 1, 2, 2, 1]],

         [[0, 0, 0, 0, 0],
          [0, 0, 1, 1, 0],
          [1, 1, 2, 2, 0],
          [3, 3, 3, 1, 1]]]).float()
    train_Y = torch.tensor([[-1, -1, -2, -2],
                            [-1, -1, -2, -2]]).float()
    train = torch.utils.data.TensorDataset(train_X, train_Y)
    train_set = torch.utils.data.DataLoader(train, batch_size=batchSize)

    vectorNet = VectorNetWithPredicting(len=len, timeStampNumber=timeStampNumber)
    vectorNet = vectorNet.to(device)
    # print(vectorNet)
    # raise NotImplementedError
    lossfunc = torch.nn.MSELoss()

    for iterator in range(epoch):
        if iterator % 5 == 0:
            # learningRate *= 0.7
            optimizer = torch.optim.Adam(vectorNet.parameters(), lr=learningRate)
        for data, target in train_set:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            # print("in")
            outputs = vectorNet(data)
            # print("out")
            loss = lossfunc(outputs, target).to(device)
            print('outputs:',outputs)
            print('target:',target)
            # print(loss.device)
            loss.backward()
            optimizer.step()
            # print("???")
        print(iterator)

count = 0
def visualize(data, labels, prediction, pID):
    img_half_scale = 20
    img_resolution = 0.1
    img_half_scale = round(img_half_scale / img_resolution)
    global count
    for traj, label, pred in zip(data, labels, prediction):
        tmp = traj[0]
        traj = traj[1:, :]
        idx = tmp[0]
        # AVX = tmp[1]
        # AVY = tmp[2]
        maxX = tmp[3] /img_resolution / 100
        maxY = tmp[4] /img_resolution / 100
        traj[:, 0:3:2] *= maxX
        traj[:, 1:4:2] *= maxY
        label[0:-2:2] *= maxX
        label[1:-1:2] *= maxY
        pred[0:-2:2] *= maxX
        pred[1:-1:2] *= maxY
        # traj *= img_resolution
        # label *= img_resolution
        traj = traj.astype("int")
        label = label.astype("int")
        pred = pred.astype("int")
        img = np.ones((img_half_scale * 2, img_half_scale * 2, 3)) * 255
        for i in range(traj.shape[0] - 2, -1, -1):
            if traj[i, 0] == 0 and traj[i + 1, 0] == 0:
                continue
            if traj[i, 4] == 2:
                line_color = (0, 0, 0)
                # img = cv2.circle(img, (traj[i, 0] + img_half_scale, traj[i, 1] + img_half_scale), 1, line_color)
                if pID[i] == pID[i + 1] and traj[i + 1, 0] != 0:
                    img = cv2.line(img, (traj[i, 0] + img_half_scale, traj[i, 1] + img_half_scale),
                                    (traj[i + 1, 0] + img_half_scale, traj[i + 1, 1] + img_half_scale), line_color, thickness=1)
            else:
                if pID[i] == idx:
                    line_color = (255, 0, 0)
                    if pID[i] == pID[i + 1] and traj[i + 1, 0] != 0:
                        img = cv2.line(img, (traj[i, 0] + img_half_scale, traj[i, 1] + img_half_scale),
                                        (traj[i + 1, 0] + img_half_scale, traj[i + 1, 1] + img_half_scale), line_color, thickness=2)
                    # img = cv2.circle(img, (traj[i, 0] + img_half_scale, traj[i, 1] + img_half_scale), 1, line_color, lineType=8)
                else:
                    line_color = (0, 255, 0)
                    img = cv2.circle(img, (traj[i, 0] + img_half_scale, traj[i, 1] + img_half_scale), 1, line_color, thickness= -1)
            

        line_color = (255, 0, 0)
        for j in range(label.shape[0] // 2 - 1):
            img = cv2.circle(img, (label[j * 2] + img_half_scale, label[j * 2 + 1] + img_half_scale), 3, line_color, thickness= 2)
        
        line_color = (0, 0, 255)
        for j in range(pred.shape[0] // 2 - 1):
            img = cv2.circle(img, (pred[j * 2] + img_half_scale, pred[j * 2 + 1] + img_half_scale), 3, line_color, thickness= -1)

            # img = cv2.line(img, (label[j * 2] + img_half_scale, label[j * 2 + 1] + img_half_scale), 
            #                 (label[j * 2 + 2] + img_half_scale, label[j * 2 + 3] + img_half_scale), line_color, thickness=2)
        cv2.imshow("img", img)
        cv2.waitKey(1)
        if config.save_view:
            cv2.imwrite(os.path.join(config.save_view_path, str(count) + ".jpg"), img)
            count += 1


def train(epoch, learningRate, batchSize):
    r"""
    Training and saving my model.
    :param epoch:
    :param learningRate:
    :param batchSize:
    :return: None
    """

    print("train data path: [%s], train data size [%d]"%(TRAIN_DATA_PATH, len(TRAIN_FILE)))
    train = ArgoDataset(TRAIN_DATA_PATH, TRAIN_FILE)
    print("test data path: [%s], test data size [%d]"%(TEST_DATA_PATH, len(TEST_FILE)))
    trainset = torch.utils.data.DataLoader(train, batch_size=batchSize, shuffle=True, drop_last=False, collate_fn=my_collate_fn)
    test = ArgoDataset(TEST_DATA_PATH, TEST_FILE)
    testset = torch.utils.data.DataLoader(test, batch_size=batchSize, collate_fn=my_collate_fn)
    if config.load_model:
        try:
            print("loading model from [%s]"%config.load_model_path)
            vectorNet = torch.load(config.load_model_path)
        except:
            print("model path error.")
    else:
        vectorNet = VectorNetWithPredicting(len=9, timeStampNumber=30)

    # for para in vectorNet.named_parameters():
    #     print(para)
    # exit(0)

    vectorNet = vectorNet.to(device)

    lossfunc = torch.nn.MSELoss()
    maelossfunc = torch.nn.L1Loss()
    lr = learningRate
    optimizer = torch.optim.Adam(vectorNet.parameters(), lr=lr)
    loss_meter = meter.AverageValueMeter()
    mae_loss_meter = meter.AverageValueMeter()
    pre_loss = float("inf")
    for iterator in range(epoch):
        print("epoch [%d]: learning rate [%.10f]"%(iterator, lr))
        loss_meter.reset()
        for ii, (data, target) in tqdm(enumerate(trainset)):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            offset = data[:, -1, :]  # [0, 0, 0, 0, 0, maxX, maxY, ..., 0]
            data = data[:, 0:data.shape[1] - 1, :]
            pID = data[0, 1:, -1].detach().cpu().numpy().copy()
            
            outputs = vectorNet(data) # [batch size, len*2]
            loss = lossfunc(outputs, torch.tanh(target))
            loss_meter.add(loss.item())
            loss.backward()
            # print(iterator)
            # print('loss=',loss.item())
            # t = askADE(outputs, target)
            # print('minDis=', t[0].item(), 'ADE=', t[1].item())
            optimizer.step()
            # data[:, :, -1] = pID
            outputs = torch.atanh(outputs.detach())
            if config.visual:
                visualize(data.detach().cpu().numpy(), target.detach().cpu().numpy(), outputs.cpu().numpy(), pID)
            if ii % 200 == 0 and ii > 0:
                print("iterate [%d], train loss: [%f]"%(ii, loss_meter.value()[0]))
            for i in range(0, outputs.shape[1], 2):
                outputs[:, i] *= offset[:, 5] / 100
                target[:, i] *= offset[:, 5] / 100
                outputs[:, i + 1] *= offset[:, 6] / 100
                target[:, i + 1] *= offset[:, 6] / 100
            maeloss = maelossfunc(outputs, target)
            mae_loss_meter.add(maeloss.item())
        print("train loss: ", loss_meter.value()[0])
        print("train mae loss: ", mae_loss_meter.value()[0])
        if loss_meter.value()[0] < pre_loss:
            pre_loss = loss_meter.value()[0]
            model_name = time.strftime(config.model_save_prefix + '%m%d_%H:%M:%S.model')
            torch.save(vectorNet, os.path.join(config.model_save_path, model_name))
        else:
            lr *= config.lr_decay
            if lr < 1e-8:
                break
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if (iterator + 1) % 5 == 0:
            # lr *= 0.3
            loss_meter.reset()
            optimizer = torch.optim.Adam(vectorNet.parameters(), lr=lr)

            minADE = torch.zeros(1).to(device)
            minDis = torch.zeros(1).to(device)
            minDis[0] = float('inf')
            for data, target in testset:
                data = data.to(device)
                target = target.to(device)

                offset = data[:, -1, :]  # [0, 0, 0, 0, 0, maxX, maxY, ..., 0]
                data = data[:, 0:data.shape[1] - 1, :]

                outputs = vectorNet(data)


                # print(outputs)
                # print(target)
                loss = lossfunc(outputs, target)
                # print('loss=', loss.item())
                loss_meter.add(loss.item())
                outputs = torch.atanh(outputs.detach())
                for i in range(0, outputs.shape[1], 2):
                    outputs[:, i] *= offset[:, 5] / 100
                    target[:, i] *= offset[:, 5] / 100
                    outputs[:, i + 1] *= offset[:, 6] / 100
                    target[:, i + 1] *= offset[:, 6] / 100
                maeloss = maelossfunc(outputs, target)
                mae_loss_meter.add(maeloss.item())
            print("test loss: ", loss_meter.value()[0])
            
            #     t = askADE(outputs, target)
            #     print('minDis=', t[0].item(), 'ADE=', t[1].item())

            #     # minADE += t[1]
            #     if minDis > t[0]:
            #         minDis = t[0]
            #         minADE = t[1]
            # print('minDis=', minDis.item(), 'minADE=', minADE.item())

    # torch.save(vectorNet, 'VectorNet-test.model')
    model_name = time.strftime(config.model_save_prefix + '%m%d_%H:%M:%S.model')
    torch.save(vectorNet, os.path.join(config.model_save_path, model_name))


def test(batchSize):
    r"""
    Test my model from file.
    :return: None
    """
    train = ArgoDataset(TRAIN_DATA_PATH, TRAIN_FILE[:500])
    trainset = torch.utils.data.DataLoader(train, batch_size=batchSize, shuffle=True, drop_last=False, collate_fn=my_collate_fn)
    test = ArgoDataset(TEST_DATA_PATH, TEST_FILE)
    testset = torch.utils.data.DataLoader(test, batch_size=batchSize, collate_fn=my_collate_fn)
    config.save_view = True
    torch.nn.Module.dump_patches = True
    try:
        print("loading model: [%s]"%config.load_model_path)
        vectorNet = torch.load(config.load_model_path)
    except:
        print("model path not set!")
        return
    vectorNet = vectorNet.to(device)


    lossfunc = torch.nn.MSELoss()
    minADE = torch.zeros(1).to(device)
    minDis = torch.zeros(1).to(device)
    minDis[0] = -1
    lossfunc = torch.nn.MSELoss()
    maelossfunc = torch.nn.L1Loss()
    loss_meter = meter.AverageValueMeter()
    mae_loss_meter = meter.AverageValueMeter()
    loss_meter.reset()
    for ii, (data, target) in tqdm(enumerate(trainset)):
        data = data.to(device)
        target = target.to(device)
        offset = data[:, -1, :]  # [0, 0, 0, 0, 0, maxX, maxY, ..., 0]
        data = data[:, 0:data.shape[1] - 1, :]
        pID = data[0, 1:, -1].detach().cpu().numpy().copy()

        outputs = vectorNet(data) # [batch size, len*2]
        loss = lossfunc(outputs, torch.tanh(target))
        loss_meter.add(loss.item())
        
        # loss.backward()
        # print(iterator)
        # print('loss=',loss.item())
        # t = askADE(outputs, target)
        # print('minDis=', t[0].item(), 'ADE=', t[1].item())
        # optimizer.step()
        # data[:, :, -1] = pID
        outputs = torch.atanh(outputs.detach())
        if config.visual:
            visualize(data.detach().cpu().numpy(), target.detach().cpu().numpy(), outputs.cpu().numpy(), pID)
        # if ii % 200 == 0 and ii > 0:
        #     print("iterate [%d], train loss: [%f]"%(ii, loss_meter.value()[0]))
        for i in range(0, outputs.shape[1], 2):
            outputs[:, i] *= offset[:, 5] / 100
            target[:, i] *= offset[:, 5] / 100
            outputs[:, i + 1] *= offset[:, 6] / 100
            target[:, i + 1] *= offset[:, 6] / 100
        maeloss = maelossfunc(outputs, target)
        mae_loss_meter.add(maeloss.item())
    print("test loss: ", loss_meter.value()[0])
    print("test mae loss: ", mae_loss_meter.value()[0])


def testCV(batchSize):
    test = load_test()
    testset = torch.utils.data.DataLoader(test, batch_size=batchSize)

    lossfunc = torch.nn.MSELoss()
    minADE = torch.zeros(1).to(device)
    minDis = torch.zeros(1).to(device)
    minDis[0] = float('inf')
    for data, target in testset:
        data = data.to(device) # [batch size, vNumber, len]
        target = target.to(device)

        # print(data.shape)
        # print(target.shape)

        outputs = torch.zeros_like(target).to(device)
        for i in range(data.shape[0]):
            vx, vy, t = 0, 0, 0
            lx, ly = 0, 0
            for j in range(data.shape[1]):
                if data[i, j, 4] == 0:
                    vx += data[i, j, 2] - data[i, j, 0]
                    vy += data[i, j, 3] - data[i, j, 1]
                    lx, ly = data[i, j, 2], data[i, j, 3]
                    t += 1
            vx = vx / t
            vy = vy / t
            for j in range(30):
                lx += vx
                ly += vy
                outputs[i, j*2] = lx
                outputs[i, j*2+1] = ly

        loss = lossfunc(outputs, target)
        # print("-----------")
        # print(outputs)
        # print(target)
        t = askADE(outputs, target)
        print('loss=', loss.item(), 'minDis=', t[0].item(), 'ADE=', t[1].item())

        if minDis > t[0]:
            minDis = t[0]
            minADE = t[1]
    print('minDis=', minDis.item(), 'minADE=', minADE.item())
    # ade += t
    # ade /= len(testXList)
    # print(ade)


if __name__ == '__main__':
    r"""
    Main function, include the parameters' building and a simple train.
    All default parameters are set to the values of paper.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", dest="batch_size", default=32, type=int)
    parser.add_argument("--epoch", dest="epoch", default=25, type=int)
    parser.add_argument("--learning_rate", dest="lr", default=0.001, type=float)
    args = parser.parse_args()
    # train(epoch=config.epoch, learningRate=config.lr, batchSize=config.batch_size)
    test(batchSize=args.batch_size)
    # testCV(batchSize=args.batch_size)

    # random_train(epoch=args.epoch, learningRate=args.lr, batchSize=args.batch_size)


"""
0.001:
loss= tensor(174.4507, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([18.6789], device='cuda:0', grad_fn=<DivBackward0>)
loss= tensor(200.6239, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([20.0312], device='cuda:0', grad_fn=<DivBackward0>)
loss= tensor(0.9591, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([1.3850], device='cuda:0', grad_fn=<DivBackward0>)

0.01:
loss= tensor(171.2146, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([18.5048], device='cuda:0', grad_fn=<DivBackward0>)
loss= tensor(204.9662, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([20.2468], device='cuda:0', grad_fn=<DivBackward0>)
loss= tensor(0.7863, device='cuda:0', grad_fn=<MseLossBackward>)
ADE= tensor([1.2541], device='cuda:0', grad_fn=<DivBackward0>)
tensor([13.3352], device='cuda:0', grad_fn=<DivBackward0>)

"""
