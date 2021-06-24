import torch.nn as nn
from torchvision import models
import torch
import torch.nn.init as init


class hashing_net(nn.Module):
    """
    Return the network: backbone + fully connected hashing layer (FCH)
    output dim. : batch_size * bit
    """
    def __init__(self, model_name, bit):
        super(hashing_net, self).__init__()
        self.model_name = model_name
        self.bit = bit

    def create_model(self):

        if self.model_name =='google':
            model = models.googlenet(pretrained=True, progress=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.bit)

        elif self.model_name =='vgg':
            model = models.vgg19_bn(pretrained=True, progress=True)
            model.classifier[6] = nn.Linear(4096, self.bit)
        else:
            model = models.resnet50(pretrained=True, progress=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.bit)
        if self.model_name == 'vgg':
            init.xavier_normal_(model.classifier[6].weight.data)
            init.constant_(model.classifier[6].bias.data, 0.0)
        else:
            init.xavier_normal_(model.fc.weight.data)
            init.constant_(model.fc.bias.data, 0.0)
        return model


if __name__=='__main__':

    model = hashing_net('res50', bit=32).create_model()
    fake_data = torch.randn(2, 3, 224, 224)
    output = model(fake_data)
    print(output.shape)
