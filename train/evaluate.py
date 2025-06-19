import torch
from torch.nn.utils.rnn import pack_padded_sequence
from utils.metrics import evaluate_metrics
from utils.captions import generate_caption

def evaluate_model(model, test_loader, vocab, criterion, device):
    model.eval()
    test_loss = 0
    gts = {}
    res = {}

    with torch.no_grad():
        for i,(images, captions, lengths) in enumerate(test_loader):
            images = images.to(device)
            captions = captions.to(device)

            adjusted_lengths = [l - 1 for l in lengths]
            # ---------- Compute Loss ----------
            targets = pack_padded_sequence(
                captions[:, 1:], adjusted_lengths,
                batch_first=True, enforce_sorted=False
            )[0]

            outputs = model(images, captions[:, :-1], adjusted_lengths)

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # ---------- Compute Captions for Metrics ----------
            for j in range(images.size(0)):
                image = images[j]
                image_id = i * test_loader.batch_size + j #Compute Unique Image ID

                # Ground truth (only the first caption is used per image)
                gt_caption_ids = captions[j].tolist()
                gt_caption = vocab.decode_caption(gt_caption_ids)
                gts[image_id] = [gt_caption]  # Wrap in list as required

                # Model prediction
                predicted_caption = generate_caption(model, image, vocab, device)
                res[image_id] = [predicted_caption]

    # ---------- Compute Evaluation Metrics ----------
    metrics = evaluate_metrics(gts, res)

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print("--- Evaluation Metrics ---")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    return avg_test_loss, metrics
