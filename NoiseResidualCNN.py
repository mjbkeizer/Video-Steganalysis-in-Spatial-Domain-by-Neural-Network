import torch.nn as nn
import torch.nn.functional as F


class PTLU(nn.Module):

    def __init__(self, in_features, T=float(7)):
        super(PTLU, self).__init__()
        self.in_features = in_features
        self.T = T

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        return F.hardtanh(x, -self.T, self.T)


class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=(3, 3), stride=1, padding=1,
                      padding_mode='zeros'),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=(3, 3), stride=1, padding=1,
                      padding_mode='zeros'),
            nn.BatchNorm2d(channel_num),
        )
        self.relu = nn.ReLU()

    def forward(self, f):
        x = f
        f = self.conv_block1(f)
        f = self.conv_block2(f)
        f = x - f  # x-f(x)
        out = self.relu(f)
        return out


class CNN_Steganalysis(nn.Module):
    def __init__(self):
        super(CNN_Steganalysis, self).__init__()

        self.ResConv = nn.Conv2d(in_channels=1, out_channels=34, kernel_size=(5, 5), stride=1)
        self.Conv1 = nn.Conv2d(in_channels=34, out_channels=34, kernel_size=(3, 3), stride=1)
        self.Conv2 = nn.Conv2d(in_channels=34, out_channels=34, kernel_size=(3, 3), stride=1)
        self.Conv3 = nn.Conv2d(in_channels=34, out_channels=34, kernel_size=(3, 3), stride=1)
        self.Conv4 = nn.Conv2d(in_channels=34, out_channels=32, kernel_size=(3, 3), stride=1)
        self.Conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1)
        self.Conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=3)

        self.ResConv_bn = nn.BatchNorm2d(34)
        self.Conv1_bn = nn.BatchNorm2d(34)
        self.Conv2_bn = nn.BatchNorm2d(34)
        self.Conv3_bn = nn.BatchNorm2d(34)
        self.Conv4_bn = nn.BatchNorm2d(32)
        self.Conv5_bn = nn.BatchNorm2d(16)
        self.Conv6_bn = nn.BatchNorm2d(16)

        self.ResConv_ptlu = PTLU(34, T=3)

        self.AvgPool1 = nn.AvgPool2d((2, 2), 2)
        self.AvgPool2 = nn.AvgPool2d((3, 3), 2)
        self.AvgPool3 = nn.AvgPool2d((3, 3), 2)
        self.AvgPool4 = nn.AvgPool2d((2, 2), 2)

        self.ResBlock1 = ResidualBlock(34)
        self.ResBlock2 = ResidualBlock(34)

        self.FullyConnected = nn.Linear(16 * 3 * 3, 3)

    def forward(self, x):
        x = self.ResConv(x)
        x = self.ResConv_ptlu(self.ResConv_bn(x))

        x = self.Conv1(x)
        x = F.relu(self.Conv1_bn(x))

        x = self.Conv2(x)
        x = F.relu(self.Conv2_bn(x))

        x = self.Conv3(x)
        x = F.relu(self.Conv3_bn(x))

        x = self.AvgPool1(x)

        x = self.ResBlock1(x)

        x = self.AvgPool2(x)

        x = self.ResBlock2(x)

        x = self.AvgPool3(x)

        x = self.Conv4(x)
        x = F.relu(self.Conv4_bn(x))

        x = self.AvgPool4(x)

        x = self.Conv5(x)
        x = F.relu(self.Conv5_bn(x))

        x = self.Conv6(x)
        x = F.relu(self.Conv6_bn(x))

        x = x.view(-1, 16 * 3 * 3)  # transform

        x = self.FullyConnected(x)

        return F.softmax(x, dim=1)
