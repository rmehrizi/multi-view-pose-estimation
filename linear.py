# Rahil Mehrizi, Oct 2018
"""Creating Model to Predict 3D Poses from Multi-view 2D Joints"""

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

class LinearModel(nn.Module):
    def __init__(self, linear_size, num_joints):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.num_joints = num_joints

        self.fc1 = nn.Linear(2*(self.num_joints), self.linear_size) 
        self.bn1 = nn.BatchNorm1d(self.linear_size)

        self.fc2 = nn.Linear(self.linear_size, self.linear_size) 
        self.bn2 = nn.BatchNorm1d(self.linear_size)

        self.fc3 = nn.Linear(self.linear_size, self.linear_size) 
        self.bn3 = nn.BatchNorm1d(self.linear_size)

        self.fc4 = nn.Linear(self.linear_size, self.linear_size) 
        self.bn4 = nn.BatchNorm1d(self.linear_size)

        self.fc5 = nn.Linear(self.linear_size, self.linear_size) 
        self.bn5 = nn.BatchNorm1d(self.linear_size)

        self.fc6 = nn.Linear(self.linear_size, 3*self.num_joints) 

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()  

        """Kaiming Initialization""" 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                std1 = math.sqrt(2. / (m.in_features*m.out_features))
                m.weight.data.normal_(0,std1).clamp_(min=-2*std1,max=2*std1) 
                std2 = math.sqrt(2. / (m.out_features))
                m.bias.data.normal_(0,std2).clamp_(min=-2*std2,max=2*std2) 

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
 
        #first block
        xin1 = x

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop(x)

        x  = x + xin1

        #second block
        xin2 = x

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.drop(x)

        x  = x + xin2

        x = self.fc6(x)

        x = x.view(x.size(0), self.num_joints, 3)
 
        return x


def MakeLinearModel(linear_size, num_joints):
    model = LinearModel(linear_size, num_joints) 
    return model
