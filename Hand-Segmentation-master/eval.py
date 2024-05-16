import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from myloss import dice_coeff
from utils import dense_crf


def eval_net(net, dataset, gpu=False):
    tot = 0
    for i, b in enumerate(dataset):
        X = b[0]
        y = b[1]

        X = torch.FloatTensor(X).unsqueeze(0)
        y = torch.ByteTensor(y).unsqueeze(0)

        with torch.no_grad():
            if gpu:
                X = Variable(X).cuda()
                y = Variable(y).cuda()
            else:
                X = Variable(X)
                y = Variable(y)

        y_pred = net(X)

        y_pred = (F.sigmoid(y_pred) > 0.6).float()
        # y_pred = F.sigmoid(y_pred).float()

        dice = dice_coeff(y_pred, y.float()).data[0]
        tot += dice


    print("i=",i)
    return tot / i
