import os
import torch

SEED = 42
VOCAB_PATH = os.path.join("..", "extras", "vocabulary.pkl")
IMAGE_DIR = os.path.join("..", "dataset", "flickr30k_images", "Images")
CSV_PATH = os.path.join("..","dataset","flickr30k_images","results.csv")
MODEL_PATH = os.path.join("..","checkpoints","caption_model.pth")

def get_config():
    return {
        "embed_size": 256,
        "hidden_size": 512,
        "num_epochs": 20,
        "learning_rate": 1e-3,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

