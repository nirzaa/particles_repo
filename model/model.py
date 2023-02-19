from ast import Mod
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model import ResNet3d


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
            self.fc1 = nn.Linear(64 * 14 * 1, 64)
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
            y = x.view(-1, 64 * 14 * 1)
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
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 32 * 29 * 4)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

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
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 32 * 19 * 2)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_1(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 19 * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 20)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(3,1), padding=(0,1),)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,2), stride=(2,2), padding=1, )
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 32 * 19 * 2)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_120(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(64 * 14 * 3, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 120)
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

def model_2d_120_10(model_type, num_classes):
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
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 32 * 29 * 4)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_120_3(model_type, num_classes):
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
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 32 * 19 * 2)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_120_1(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 19 * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 20)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(3,1), padding=(0,1),)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,2), stride=(2,2), padding=1, )
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 32 * 19 * 2)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model


def model_2d_60(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(16 * 28 * 6, 256)
            self.fc2 = nn.Linear(256, 60)
            # self.fc3 = nn.Linear(256, 60)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=1, )
            # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # x = self.conv3(x)
            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 16 * 28 * 6)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            # x = self.fc3(F.relu(x))

            return x
    
    model = Model()
    return model

def model_2d_60_10(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(16 * 28 * 3, 256)
            self.fc2 = nn.Linear(256, 60)
            # self.fc3 = nn.Linear(32, 60)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=1, )
            # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 16 * 28 * 3)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            # x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_60_3(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 19 * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 60)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(3,3), padding=(0,1),)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,2), stride=(2,2), padding=1, )
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 32 * 19 * 2)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_60_1(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(32 * 19 * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 60)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), stride=(3,1), padding=(0,1),)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,2), stride=(2,2), padding=1, )
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 32 * 19 * 2)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model


def model_2d_48(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(16 * 28 * 6, 256)
            self.fc2 = nn.Linear(256, 48)
            # self.fc3 = nn.Linear(256, 60)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=1, )
            # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # x = self.conv3(x)
            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 16 * 28 * 6)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            # x = self.fc3(F.relu(x))

            return x
    
    model = Model()
    return model

def model_2d_48_10(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(16 * 28 * 3, 256)
            self.fc2 = nn.Linear(256, 48)
            # self.fc3 = nn.Linear(32, 60)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=1, )
            # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 16 * 28 * 3)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            # x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_48_5(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(16 * 28 * 2, 256)
            self.fc2 = nn.Linear(256, 48)
            # self.fc3 = nn.Linear(32, 60)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=1, )
            # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 16 * 28 * 2)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            # x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model

def model_2d_48_1(model_type, num_classes):
    class Model(nn.Module):
        '''
        The semi fully conventional architecture of the neural net
        '''
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(16 * 28 * 1, 256)
            self.fc2 = nn.Linear(256, 48)
            # self.fc3 = nn.Linear(32, 60)
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2,2), stride=(2,2), padding=1,)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=1, )
            # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1, )
            self.global1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)

            # x = self.conv3(x) <-------- too small for this

            # x = self.global1(x)
            # x = self.avgpool(x)
            y = x.view(-1, 16 * 28 * 1)
            # x = x.view(-1, 2 * 2 * 2)
            x = self.fc1(F.relu(y))
            x = self.fc2(F.relu(x))
            # x = self.fc3(F.relu(x))

            return x
    model = Model()
    return model