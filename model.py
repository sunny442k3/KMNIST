import sys
import torch
import torchvision
import torch.nn as nn
from decoder import MLDecoder


class AlexNet(nn.Module):
    def __init__(self, cf):
        super(AlexNet, self).__init__()
        
        model = torchvision.models.alexnet(weights="IMAGENET1K_V1")
        model.features[0] = torch.nn.Conv2d(cf.input_dim, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False)
        self.logit = model 
        self.fc = nn.Linear(1000, cf.num_classes, bias=True)

    def forward(self, x):
        out = self.logit(x)
        return self.fc(out)

class ResNet(nn.Module):

    def __init__(self, cf):
        super(ResNet, self).__init__()
        model = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        model.conv1 = torch.nn.Conv2d(cf.input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.logit = model
        self.fc = nn.Linear(1000, cf.num_classes, bias=True)

    def forward(self, x):
        out = self.logit(x)
        return self.fc(out)

def Backbone(backbone):
    backbone_dict = {"alexnet": AlexNet, "resnet": ResNet}
    return backbone_dict[backbone.lower()]

class CNNModelOptimal(nn.Module):
    def __init__(self, cf):
        super(CNNModelOptimal, self).__init__()
        backbone = Backbone(cf.backbone)
        if len(cf.load_backbone_weight.strip()):
            backbone_weight = torch.load(cf.load_backbone_weight.strip())["model"]
            backbone.load_state_dict(backbone_weight)
        # backbone = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        # backbone.conv1 = torch.nn.Conv2d(cf.input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.dropout = nn.Dropout(p=cf.dropout)
        self.decoder = MLDecoder(num_classes=cf.num_classes, query_weight=cf.query_weight)

    def forward(self, x):
        out = self.backbone(x)
        # print(out.size())
        # sys.exit()
        out = self.dropout(out)
        out = self.decoder(out)
        return out