import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model = resnet

        modules=list(resnet.children())[:-1]
        self.main = nn.Sequential(*modules)
        # take the last layer representation from the resnet
        self.out_dim = resnet.fc.in_features

    def forward(self, x):
        x = self.main(x)

        return x.squeeze(-1).squeeze(-1)

    def domain_features(self, x):
        '''
        get domain features for dg_mmld
        '''
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        return x.view(x.size(0), -1)

    def conv_features(self, x) :
        '''
        get domain features for dg_mmld
        '''
        results = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        # results.append(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        results.append(x)
        x = self.base_model.layer2(x)
        results.append(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        # results.append(x)
        return results
