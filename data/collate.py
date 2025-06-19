import torch
from torch.nn.utils.rnn import pad_sequence

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)

        #pad sequences 
        lengths = [len(caption) for caption in captions]
        captions_padded = pad_sequence(captions, batch_first = True, padding_value=self.pad_idx)

        return torch.stack(images), captions_padded, torch.tensor(lengths)