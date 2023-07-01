import torch
import torchvision
import torch.nn as nn

class ResNet(nn.Module):

    def __init__(self, cf):
        super(ResNet, self).__init__()

        model = torchvision.models.resnet101(weights=True)
        model.conv1 = torch.nn.Conv2d(cf.input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.logit = model
        self.fc = nn.Linear(1000, cf.num_classes, bias=True)
    
    def forward(self, x):
        out = self.logit(x)
        return self.fc(out)