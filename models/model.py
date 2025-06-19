import torch.nn as nn

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, lengths):
        features = self.encoder(images)  # (B, embed_size)
        outputs = self.decoder(features, captions, lengths)
        return outputs
