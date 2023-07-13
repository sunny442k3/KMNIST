import torch.nn as nn
from decoder import MLDecoder

class BasicBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_sizes, strides, paddings):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0], bias=False)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_dim, input_dim, kernel_size=kernel_sizes[1], stride=strides[1], padding=paddings[1])
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
    
class ResCNN(nn.Module):

    def __init__(self):
        super(ResCNN, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
             nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        )
        self.cnn_block1 = BasicBlock(64, 64, [5, 3], [1, 1], [2, 1]),
        self.switch1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.cnn_block2 = BasicBlock(64, 64, [5, 3], [1, 1], [2, 1])
        self.switch2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.cnn_block3 = BasicBlock(128, 128, [3, 3], [1, 1], [1, 1])

    def forward(self, x):
        out = self.input_block(x)
        out1 = self.cnn_block1(out)
        out1 = self.switch1(out1)
        out2 = self.cnn_block2(out1)
        out2 = self.switch2(out2)
        return self.cnn_block3(out2)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=False)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=False)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.cnn_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False)
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        return x

def Backbone(cf):
    if "cnn" == cf.backbone:
        model = CNN()
        return model
    model = ResCNN()
    return model

class CNNModelOptimal(nn.Module):
    def __init__(self, cf):
        super(CNNModelOptimal, self).__init__()
        self.backbone = Backbone(cf)
        self.decoder = MLDecoder(
            num_classes=cf.num_classes, 
            query_weight=cf.query_weight, 
            hidden_dim=128, 
            freeze_query=cf.freeze_query
        )

    
    def forward(self, x):
        out = self.backbone(x)
        out = self.decoder(out)
        return out