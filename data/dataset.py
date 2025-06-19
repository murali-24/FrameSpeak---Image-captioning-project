import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

class Flickr30kDataset(Dataset):

    def __init__(self, image_dir, caption_dict, vocab, transform=None):
        self.image_dir = image_dir
        self.caption_dict = list(caption_dict.items()) # [(image_name, [captions])]
        self.vocab = vocab #dictionary of all words used with index allocated for each word
        self.transform = transform #transformations to be applied for images

    def __len__(self):#returns number of images present
        return len(self.caption_dict)

    def __getitem__(self, idx): #What Happens on dataset[i]
        image_name, captions = self.caption_dict[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")#to ensure image has 3 channels

        #one caption is chosen at random, but over many epoch all captions will be choosed
        caption = random.choice(captions) 

        #tokenization words into numerical tensors
        tokens = [self.vocab["<start>"]]
        tokens += [self.vocab[word] for word in caption.split()]
        tokens += [self.vocab["<end>"]]

        #apply image transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(tokens)
