import torch
from torch.nn.utils.rnn import pack_padded_sequence
from utils.metrics import evaluate_metrics
from utils.captions import generate_caption

def train_model(model, train_loader,val_loader, criterion, optimizer, vocab, num_epochs, device, clip_value=10):
    model.to(device=device)

    for epoch in range(1, num_epochs+1):

        # ---------------------- TRAINING ----------------------
        model.train() #to make model run in train mode
        train_loss = 0 #used to track average loss per epoch

        for batch_idx, (image_batch, captions, lengths) in enumerate(train_loader):
            image_batch = image_batch.to(device)
            captions = captions.to(device)

            adjusted_lengths = [l-1 for l in lengths]
            #Targets: all words except the first one <start>
            targets = pack_padded_sequence(input=captions[:,1:],
                                                        lengths=adjusted_lengths,
                                                        batch_first=True,
                                                        enforce_sorted=False,
                                                        )[0]
            #Outputs: generated from models
            outputs = model(image_batch, captions[:, :-1], torch.tensor(adjusted_lengths))

            #compute loss
            loss = criterion(outputs, targets)

            #reset gradient to zero
            optimizer.zero_grad()

            #compute gradients
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            #Update weights based on gradients
            optimizer.step()

            train_loss += loss.item() #accumulate epoch loss

            #batch loss
            if (batch_idx ) % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        #average loss in an epoch     
        avg_train_loss = train_loss / len(train_loader)


        # ---------------------- VALIDATION ----------------------
        model.eval()
        val_loss = 0
        gts = {}
        res = {}

        with torch.no_grad():
            for i,(images, captions, lengths) in enumerate(val_loader):
                images = images.to(device)
                captions = captions.to(device)
                adjusted_lengths = [l-1 for l in lengths]
                # ---------- Compute Loss ----------
                targets = pack_padded_sequence(
                    captions[:, 1:], adjusted_lengths,
                    batch_first=True, enforce_sorted=False
                )[0]

                outputs = model(images, captions[:, :-1], adjusted_lengths)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                
                # ---------- Compute Captions for Metrics ----------
                for j in range(images.size(0)):
                    image = images[j]
                    image_id = i * val_loader.batch_size + j #Compute Unique Image ID

                    # Ground truth (only the first caption is used per image)
                    gt_caption_ids = captions[j].tolist()
                    gt_caption = vocab.decode_caption(gt_caption_ids)
                    gts[image_id] = [gt_caption]  # Wrap in list as required

                    # Model prediction
                    predicted_caption = generate_caption(model, image, vocab, device)

                    if predicted_caption and isinstance(predicted_caption, str) and predicted_caption.strip():
                        res[image_id] = [predicted_caption]
                    else:
                        res[image_id] = ["<unk>"]  # fallback safe caption
                    
        avg_val_loss = val_loss / len(val_loader)
            
        # ---------------------- SUMMARY ----------------------
        print(f"\nEpoch [{epoch}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss  : {avg_val_loss:.4f}\n")
        if (epoch==1) or (epoch%4 == 0):
            print("--- Evaluation Metrics ---")
            metrics = evaluate_metrics(gts, res)
            for metric, score in metrics.items():
                print(f"{metric}: {score:.4f}")

