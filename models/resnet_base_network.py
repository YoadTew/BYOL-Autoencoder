import torchvision.models as models
from models.resnet import resnet18, resnet50
import torch
from models.mlp_head import MLPHead


class ResNet18(torch.nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = resnet18(pretrained=False, in_channels=in_channels)
        elif kwargs['name'] == 'resnet50':
            resnet = resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
