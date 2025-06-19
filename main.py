import pickle
import os
import torch.nn as nn
import torch.optim as optim
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data.dataset import Flickr30kDataset
from data.collate import Collate
from data.vocab import Vocabulary
from data.preprocess import load_and_process_captions
from utils.transforms import get_transform
from utils.config import SEED, IMAGE_DIR, VOCAB_PATH,CSV_PATH,MODEL_PATH,get_config
from models.model import ImageCaptioningModel
from models.decoder import DecoderRNN
from models.encoder import EncoderCNN
from train.train import train_model
from train.evaluate import evaluate_model

csv_path = CSV_PATH
caption_dict = load_and_process_captions(csv_path)

# 1. Load captions and split
items = sorted(caption_dict.items(), key=lambda x: x[0])
train_set, temp_set = train_test_split(items, test_size=0.25, random_state=SEED)
val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=SEED)

train_dict, val_dict, test_dict = map(dict, [train_set, val_set, test_set])

# 2. Vocabulary
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
else:
    vocab = Vocabulary()
    vocab.build_vocab(caption_dict, threshold=4)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

# 3. Loaders
transform = get_transform()
collator = Collate(pad_idx=vocab["<pad>"])
config = get_config()

train_loader = DataLoader(Flickr30kDataset(IMAGE_DIR, train_dict, vocab, transform), batch_size=32, shuffle=True, collate_fn=collator)
val_loader = DataLoader(Flickr30kDataset(IMAGE_DIR, val_dict, vocab, transform), batch_size=32, shuffle=False, collate_fn=collator)
test_loader = DataLoader(Flickr30kDataset(IMAGE_DIR, test_dict, vocab, transform), batch_size=32, shuffle=False, collate_fn=collator)

encoder = EncoderCNN(embed_size=config["embed_size"])
decoder = DecoderRNN(embed_size=config["embed_size"],
                        hidden_size=config["hidden_size"],
                        vocab_size=len(vocab))

model = ImageCaptioningModel(encoder, decoder)

# 4. Loss & Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

train_model(model,train_loader,val_loader,criterion,optimizer,vocab,config["num_epochs"],
            config["device"],clip_value=10)

# Save
torch.save(model.state_dict(), MODEL_PATH)

# # Load
# model.load_state_dict(torch.load(MODEL_PATH))
# model.to(config["device"])
evaluate_model(model,test_loader,vocab,criterion,device=config["device"],)
