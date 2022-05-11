from ast import Mod
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model import ResNet3d
import torch


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def general_model(model_type, num_classes):
    smp_size = 16 * 5
    smp_dur = 64

    if model_type == "ResNet10":
        model = ResNet3d.ResNet(ResNet3d.BasicBlock, [1, 1, 1, 1], sample_size=smp_size,
                                  sample_duration=smp_dur, num_classes=num_classes)

    if model_type == 'ResNet34':
        model = ResNet3d.ResNet(ResNet3d.BasicBlock, [3, 4, 6, 3], sample_size=smp_size,
                                  sample_duration=smp_dur, num_classes=num_classes)

    return model

def model_2d(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(64 * 14 * 3, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 20)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 64 * 14 * 3)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    
    model = Model()
    return model

def model_2d_10(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 29 * 4, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 20)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=1, )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            y = x.view(-1, 32 * 29 * 4)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

# def model_2d_3(model_type, num_classes):
#     class Model(nn.Module):
#         '''
#         The semi fully conventional architecture of the neural net
#         '''
#         def __init__(self):
#             super(Model, self).__init__()
#             self.fc1 = nn.Linear(32 * 37 * 3, 64)
#             self.fc2 = nn.Linear(64, 32)
#             self.fc3 = nn.Linear(32, 20)
#             self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=3,)

#         def forward(self, x):
#             y_list = list()
#             for i in range(x.shape[-1]):
#                 y = self.conv1(x[:,:,:,i])
#                 y_list.append(y)
#             y = torch.stack(y_list, axis=0)
#             y = torch.reshape(y, (y.shape[1], y.shape[2], y.shape[3], y.shape[0]))
#             y = y.view(-1, 32 * 37 * 3)
#             x = self.fc1(F.relu(y))
#             x = self.fc2(F.relu(x))
#             x = self.fc3(F.relu(x))

#             return x
#     model = Model()
#     return model

def model_2d_1(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 37 * 1, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 20)
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=3,)

        def forward(self, x):
            y_list = list()
            for i in range(x.shape[-1]):
                y = self.conv1(x[:,:,:,i])
                y_list.append(y)
            y = torch.stack(y_list, axis=0)
            y = torch.reshape(y, (y.shape[1], y.shape[2], y.shape[3], y.shape[0]))
            y = y.view(-1, 32 * 37 * 1)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

# ========= My architectures =========== #

# def model_2d_1(model_type, num_classes):
#     class Model(nn.Module):
#         '''
#         The semi fully conventional architecture of the neural net
#         '''
#         def __init__(self):
#             super(Model, self).__init__()
#             self.fc1 = nn.Linear(32 * 19 * 2, 64)
#             self.fc2 = nn.Linear(64, 32)
#             self.fc3 = nn.Linear(32, 20)
#             self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(3,1), padding=(0,1),)
#             self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,2), stride=(2,2), padding=1, )

#         def forward(self, x):
#             x = self.conv1(x)
#             x = self.conv2(x)
#             y = x.view(-1, 32 * 19 * 2)
#             x = self.fc1(F.relu(y))
#             x = self.fc2(F.relu(x))
#             x = self.fc3(F.relu(x))

#             return x
#     model = Model()
#     return model

def model_2d_3(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 19 * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 20)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(3,3), padding=(0,1),)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,2), stride=(2,2), padding=1, )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            y = x.view(-1, 32 * 19 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_3_diff(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(8 * 18 * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 20)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(3,3), padding=(0,1),)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1), stride=(1,1), padding=(0,1),)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2,3), stride=(1,1), padding=(1,0),)
            self.conv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3,3), stride=(2,1), padding=(0,1),)
            self.fc1 = nn.Linear(8 * 18 * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 20)

        def forward(self, x):
            x = self.conv1(x)
            y = self.conv2(x)
            y = F.relu(self.conv3(y))
            z = x + y
            x = self.conv4(z)
            x =  x.view(-1, 8 * 18 * 2)
            x = self.fc1(F.relu(x))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))
            

            return x
    model = Model()
    return model


def model_2d_10_100classes(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 29 * 4, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 100)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=1, )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            y = x.view(-1, 32 * 29 * 4)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model


def model_2d_10_110classes(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 29 * 4, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 110)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=1, )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            y = x.view(-1, 32 * 29 * 4)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_all_110classes(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 29 * 6, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 110)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=1, )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            y = x.view(-1, 32 * 29 * 6)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_all_60classes(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 29 * 6, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 60)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=1, )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            y = x.view(-1, 32 * 29 * 6)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_all_20classes(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 29 * 6, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 20)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=1, )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            y = x.view(-1, 32 * 29 * 6)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model
