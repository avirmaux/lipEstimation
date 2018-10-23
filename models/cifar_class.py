import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNet(nn.Module):

    def __init__(self):

        super(DeepNet, self).__init__()

        # self.nl = nn.LeakyReLU(0.3)
        self.nl = nn.ELU(0.3)

        # 32 x 32 x 3
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(96, 128, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128, 1e-3)
        self.pool1 = nn.FractionalMaxPool2d(kernel_size=3, output_size=(23, 23))
        self.dropout1 = nn.Dropout(p=0.1)

        # 23 x 23
        self.conv3 = nn.Conv2d(128, 384, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(256, 384, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(384, 1e-3)
        self.dropout2 = nn.Dropout(p=0.2)
        self.pool2 = nn.FractionalMaxPool2d(3, output_size=(16 ,16))

        # 16 x 16
        self.conv5 = nn.Conv2d(384, 768, kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(512, 768, kernel_size=1, padding=0)
        self.bn6 = nn.BatchNorm2d(768, 1e-3)
        self.dropout3 = nn.Dropout(p=0.25)
        self.pool3 = nn.FractionalMaxPool2d(3, output_size=(11, 11))

        # 11 x 11
        self.conv7 = nn.Conv2d(768, 1280, kernel_size=3, padding=1)
        # self.conv8 = nn.Conv2d(1280, 1280, kernel_size=1, padding=0)
        self.bn7 = nn.BatchNorm2d(1280, 1e-3)
        self.dropout4 = nn.Dropout(p=0.35)
        self.pool4 = nn.FractionalMaxPool2d(3, output_size=(7,7))

        # 7 x 7
        self.conv9 = nn.Conv2d(1280, 1920, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout(p=0.4)
        self.pool5 = nn.FractionalMaxPool2d(3, output_size=(4, 4))

        # 4 x 4
        self.conv10 = nn.Conv2d(1920, 2048, kernel_size=2, padding=0)

        # 3 x 3
        self.conv11 = nn.Conv2d(2048, 2560, kernel_size=2, padding=0)
        self.bn11   = nn.BatchNorm2d(2560, 1e-3)

        # 2 x 2
        self.conv12 = nn.Conv2d(2560, 3072, kernel_size=2, padding=0)
        self.dropout5 = nn.Dropout(p=0.45)
        self.bn12   = nn.BatchNorm2d(3072, 1e-3)

        # 1 x 1
        self.fc1 = nn.Linear(3072, 100)

    def forward(self, x):
        x = self.nl(self.conv1(x))
        # x = self.nl(self.conv2(x))
        x = self.pool1(x)
        x = self.bn2(x)
        x = self.dropout1(x)

        x = self.nl(self.conv3(x))
        # x = self.nl(self.conv4(x))
        x = self.pool2(x)
        x = self.bn4(x)
        x = self.dropout2(x)

        x = self.nl(self.conv5(x))
        # x = self.nl(self.conv6(x))
        x = self.pool3(x)
        x = self.bn6(x)
        x = self.dropout3(x)

        x = self.nl(self.conv7(x))
        # x = self.nl(self.conv8(x))
        x = self.pool4(x)
        x = self.bn7(x)

        x = self.nl(self.conv9(x))
        x = self.pool5(x)
        x = self.dropout4(x)
        x = self.nl(self.conv10(x))
        x = self.nl(self.conv11(x))
        x = self.bn11(x)
        x = self.nl(self.conv12(x))
        x = self.dropout5(x)
        x = self.bn12(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.elu(x, inplace=True)
