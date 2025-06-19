import torch

def generate_caption(model, image, vocab, device, max_len=20, transform=None):
    """
    Generates a caption string from a single image tensor using the model.
    """
    model.eval()
    if transform:
        image = transform(image)
    image = image.unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        # Encode image features
        features = model.encoder(image)  # (1, embed_size)

        # Generate token IDs using decoder
        start_token_idx = vocab["<start>"]
        sampled_ids = model.decoder.sample(features, start_token_idx ,max_len=max_len)  # (1, max_len)
        sampled_ids = sampled_ids[0].tolist()  # Remove batch dim → list of ints

        # Convert IDs to words
        caption = vocab.decode_caption(sampled_ids)
    
    return caption