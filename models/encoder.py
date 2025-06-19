import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):

    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        for param in resnet.parameters():
            param.requires_grad = False#freezing gradients, transfer learning   

        modules = list(resnet.children())[:-1] #removing last layer
        self.resnet = nn.Sequential(*modules) #combining remaining layers
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)#resizing to embedding size
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)  # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        features = self.bn(self.linear(features))  # (B, embed_size)
        return features