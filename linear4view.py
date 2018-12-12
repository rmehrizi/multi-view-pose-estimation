# Rahil Mehrizi, Oct 2018
"""Creating Model to Predict 3D Poses from Multi-view 2D Joints"""

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

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

        self.fc7 = nn.Linear(self.linear_size , self.linear_size) 
        self.bn7 = nn.BatchNorm1d(self.linear_size)

        self.fc8 = nn.Linear(self.linear_size, self.linear_size) 
        self.bn8 = nn.BatchNorm1d(self.linear_size)

        self.fc9 = nn.Linear(self.linear_size, self.linear_size) 
        self.bn9 = nn.BatchNorm1d(self.linear_size)

        self.fc10 = nn.Linear(self.linear_size, self.linear_size) 
        self.bn10 = nn.BatchNorm1d(self.linear_size)

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

    def forward(self, x1, x2, x3, x4, mean, sd, r1, r2, r3, r4, v1, v2, v3, v4):

        Input = [x1, x2, x3, x4]
        Rotation = [r1,r2,r3,r4]
        Output = []

        """Estimation 3D Pose for Each View Separately"""  
        for i in range(0,4):

            x=Input[i]
            r=Rotation[i]

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

            #third block
            xin3 = x

            x = self.fc7(x)
            x = self.bn7(x)
            x = self.relu(x)
            x = self.drop(x)

            x = self.fc8(x)
            x = self.bn8(x)
            x = self.relu(x)
            x = self.drop(x)

            x  = x + xin3

            #fourth block
            xin4 = x

            x = self.fc9(x)
            x = self.bn9(x)
            x = self.relu(x)
            x = self.drop(x)

            x = self.fc10(x)
            x = self.bn10(x)
            x = self.relu(x)
            x = self.drop(x)

            x  = x + xin4

            x = self.fc6(x)

            x = x.view(x.size(0), self.num_joints, 3)
  
            # transferring to the global coordinates
            x = x * sd.cuda() + mean.cuda()
            x = torch.transpose(x,1,2)
            x = torch.bmm (r , x) 
            x = torch.transpose(x,1,2)

            x = x.contiguous()
            x = x.view(x.size(0), self.num_joints, 3)
            Output.append(x)


        """Multi-view Fusion"""
        sv = v1 + v2 + v3 + v4 
        v1 = (v1)/sv
        v2 = (v2)/sv
        v3 = (v3)/sv
        v4 = (v4)/sv
  
        v1 = v1.unsqueeze(2).expand(v1.size(0), v1.size(1), 3)
        v2 = v2.unsqueeze(2).expand(v2.size(0), v2.size(1), 3)
        v3 = v3.unsqueeze(2).expand(v3.size(0), v3.size(1), 3)
        v4 = v4.unsqueeze(2).expand(v4.size(0), v4.size(1), 3)

        out = v1*Output[0] + v2*Output[1] + v3*Output[2] + v4*Output[3] 

        return out



def MakeLinearModel(linear_size, num_joints):
    model = LinearModel(linear_size, num_joints) 
    return model
