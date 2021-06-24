import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch


class back_bone(nn.Module):

    def __init__(self, model_name, bit):
        super(back_bone, self).__init__()
        self.model_name = model_name
        self.bit = bit

    def create_model(self):
        if self.model_name == 'google':

            model = models.googlenet(pretrained=True, progress=True)

            '''
            model = models.GoogLeNet(transform_input=True, aux_logits=True, init_weights=False)
            model.load_state_dict(torch.load("googlenet.pth"))
            #num_features = model.AuxLogits.fc.in_features
            #model.AuxLogits.fc = nn.Linear(num_features, self.bit)
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
            '''
            # handle main network
            num_features = model.fc.in_features     # 1024 actual.
            model.fc = nn.Linear(num_features, self.bit)

        else:
            model = models.resnet50(pretrained=True, progress=True)
            # model = models.resnet50()
            # model.load_state_dict(torch.load("resnet50-19c8e357.pth"))
            # model.load_state_dict(state_dict)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.bit)
            ########handle main network
        return model
