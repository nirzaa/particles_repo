import torch.nn as nn
import torch


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# Linear model
class Net(nn.Module):
    def __init__(self, num_cl):
        super(Net, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(25410, num_cl),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        out = self.linear_layers(out)
        return out


class SmallNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SmallNet, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool3d(kernel_size=5, divisor_override=1),
            torch.nn.Conv3d(1, 18, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
        )

        self.fc = nn.Sequential(
            torch.nn.Linear(396, 64),
            nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x


class MediumNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MediumNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(17280, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
